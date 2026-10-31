import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 100

def _score(h):
    """Compute the maximum overlap of h with shifted (1-h)."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=200.0):
    """Smooth approximation of max(x)."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _overlap_and_grad(h, alpha):
    """Compute overlap vector and its gradient w.r.t. h."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    
    # Gradient of overlap_i w.r.t. h_j
    # For i < N-1: overlap_i = h_i * (1 - h_{i+1})
    #   d(overlap_i)/dh_i = 1 - h_{i+1}
    #   d(overlap_i)/dh_{i+1} = -h_i
    # For i = N-1: overlap_{N-1} = h_{N-1}
    #   d(overlap_{N-1})/dh_{N-1} = 1
    
    grad = np.zeros(n)
    # Contribution from overlap_i for each i
    for i in range(n):
        if i < n - 1:
            grad[i] += (1.0 - h[i+1])
            grad[i+1] += (-h[i])
        else:
            grad[i] += 1.0
    
    # Smooth max gradient
    x_max = np.max(overlap)
    e = np.exp(alpha * (overlap - x_max))
    p = e / np.sum(e)
    
    grad_smooth = p * grad
    return overlap, grad_smooth

def _fix_sum(h):
    """Project h to have sum exactly N/2 while staying in [0,1]."""
    h = h.copy()
    for _ in range(50):
        diff = N / 2.0 - np.sum(h)
        if abs(diff) < 1e-14:
            break
        # Adjust elements with most slack
        slack_up = 1.0 - h
        slack_down = h
        if diff > 0:
            # Need to increase sum; prioritize elements with room to go up
            order = np.argsort(-slack_up)
            for i in order:
                if abs(diff) < 1e-14:
                    break
                amt = min(slack_up[i], diff)
                h[i] += amt
                diff -= amt
        else:
            # Need to decrease sum; prioritize elements with room to go down
            order = np.argsort(-slack_down)
            for i in order:
                if abs(diff) < 1e-14:
                    break
                amt = min(slack_down[i], -diff)
                h[i] -= amt
                diff += amt
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 35s safety margin before 150s kill
    
    h_best = np.full(N, 0.5)
    score_best = _score(h_best)
    
    # Strategy 1: Sawtooth pattern - piecewise constant with two values a, b
    # Pattern: a repeated for P1 steps, b repeated for P2 steps, periodic
    # Sum constraint: (P1*a + P2*b) * (N/(P1+P2)) = N/2 => P1*a + P2*b = (P1+P2)/2
    
    def sawtooth_params(params):
        """Generate h from sawtooth parameters.
        params: [a, b, P1, P2, phase]
        a, b in [0, 1], P1, P2 in [1, N], phase in [0, P1+P2)
        """
        a, b, P1, P2, phase = params
        P1 = int(round(P1))
        P2 = int(round(P2))
        P1 = max(1, min(N, P1))
        P2 = max(1, min(N, P2))
        period = P1 + P2
        
        # Build one period
        period_h = np.zeros(period)
        period_h[:P1] = a
        period_h[P1:] = b
        
        # Tile and shift by phase
        n_tiles = (N + period - 1) // period
        h = np.tile(period_h, n_tiles)[:N]
        
        # Apply phase shift
        if phase > 0:
            phase = int(round(phase)) % period
            h = np.roll(h, -phase)
        
        return h
    
    def sawtooth_objective(params):
        h = sawtooth_params(params)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        return _score(h)
    
    # Global search over sawtooth parameters
    bounds = [(0.0, 1.0), (0.0, 1.0), (1.0, 30.0), (1.0, 30.0), (0.0, 60.0)]
    
    if time.time() < deadline - 60:
        try:
            result = differential_evolution(
                sawtooth_objective,
                bounds=bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                polish=False,
                init='latinhypercube'
            )
            h_candidate = sawtooth_params(result.x)
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            h_candidate = _fix_sum(h_candidate)
            sc = _score(h_candidate)
            if sc < score_best:
                score_best = sc
                h_best = h_candidate.copy()
        except Exception:
            pass
    
    # Strategy 2: Local optimization using SLSQP with epigraph formulation
    # Minimize t subject to overlap_i <= t for all i, sum(h) = N/2, 0 <= h <= 1
    
    if time.time() < deadline - 30:
        def _epigraph_obj(params):
            h = params[:N]
            t = params[N]
            h = np.clip(h, 0.0, 1.0)
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            # Penalty for constraint violations
            viol = np.maximum(overlap - t, 0.0)
            return t + 10.0 * np.sum(viol**2)
        
        def _epigraph_grad(params):
            h = params[:N]
            t = params[N]
            h = np.clip(h, 0.0, 1.0)
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            
            grad = np.zeros(N + 1)
            # dt
            grad[N] = 1.0 + 20.0 * np.sum(np.maximum(overlap - t, 0.0))
            
            # dh_j
            for i in range(N):
                if i < N - 1:
                    # overlap_i = h_i * (1 - h_{i+1})
                    if overlap[i] > t:
                        pen = 20.0 * (overlap[i] - t)
                        grad[i] += pen * (1.0 - h[i+1])
                        grad[i+1] += pen * (-h[i])
                else:
                    # overlap_{N-1} = h_{N-1}
                    if overlap[i] > t:
                        pen = 20.0 * (overlap[i] - t)
                        grad[i] += pen * 1.0
            
            return grad
        
        # Initialize with current best
        h_init = h_best.copy()
        t_init = _score(h_init) + 0.05
        params_init = np.concatenate([h_init, [t_init]])
        
        bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)]
        constraints = [
            {'type': 'eq', 'fun': lambda p: np.sum(p[:N]) - N/2.0,
             'jac': lambda p: np.concatenate([np.ones(N), [0.0]])}
        ]
        
        try:
            result = minimize(
                _epigraph_obj,
                params_init,
                jac=_epigraph_grad,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            h_candidate = np.clip(result.x[:N], 0.0, 1.0)
            h_candidate = _fix_sum(h_candidate)
            sc = _score(h_candidate)
            if sc < score_best:
                score_best = sc
                h_best = h_candidate.copy()
        except Exception:
            pass
    
    # Strategy 2b: Projected gradient descent with smooth max as backup
    if time.time() < deadline - 20:
        h = h_best.copy()
        
        for alpha in [100.0, 200.0, 400.0, 800.0]:
            if time.time() > deadline - 20:
                break
            
            lr = 0.02
            stuck = 0
            for step in range(800):
                if time.time() > deadline - 20:
                    break
                
                h_clipped = np.clip(h, 0.0, 1.0)
                h_clipped = _fix_sum(h_clipped)
                
                overlap, grad_smooth = _overlap_and_grad(h_clipped, alpha)
                
                # Projected gradient step
                h_new = h_clipped - lr * grad_smooth
                h_new = np.clip(h_new, 0.0, 1.0)
                h_new = _fix_sum(h_new)
                
                sc_new = _score(h_new)
                if sc_new < score_best - 1e-12:
                    score_best = sc_new
                    h_best = h_new.copy()
                    h = h_new.copy()
                    stuck = 0
                    lr *= 0.998
                else:
                    stuck += 1
                    lr *= 0.95
                    if stuck > 80 or lr < 1e-9:
                        break
    
    # Strategy 3: Fourier-based refinement
    # Parameterize h = 0.5 + sum of Fourier modes, optimize coefficients
    
    x = np.arange(N)
    
    def fourier_h(params, K):
        h = np.full(N, 0.5)
        for k in range(K):
            h += params[2*k] * np.cos(2 * np.pi * k * x / N)
            h += params[2*k+1] * np.sin(2 * np.pi * k * x / N)
        return h
    
    def fourier_objective(params, K):
        h = fourier_h(params, K)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        return _score(h)
    
    for K in [3, 5, 8, 12]:
        if time.time() > deadline - 15:
            break
        
        n_params = 2 * K
        # Initialize from current best
        h_init = h_best
        # Project to Fourier space
        coeffs = np.zeros(n_params)
        for k in range(K):
            cos_k = np.cos(2 * np.pi * k * x / N)
            sin_k = np.sin(2 * np.pi * k * x / N)
            coeffs[2*k] = np.dot(h_init - 0.5, cos_k) * 2 / N
            coeffs[2*k+1] = np.dot(h_init - 0.5, sin_k) * 2 / N
        
        try:
            result = minimize(
                lambda p: fourier_objective(p, K),
                coeffs,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            h_candidate = fourier_h(result.x, K)
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            h_candidate = _fix_sum(h_candidate)
            sc = _score(h_candidate)
            if sc < score_best:
                score_best = sc
                h_best = h_candidate.copy()
        except Exception:
            pass
    
    # Final high-precision polish: SLSQP with tight tolerances
    if time.time() < deadline - 5:
        def _polish_obj(h):
            return _score(h)
        
        def _polish_grad(h):
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros(N)
            for i in range(N):
                h_plus = h.copy()
                h_plus[i] += eps
                h_plus = np.clip(h_plus, 0.0, 1.0)
                h_plus = _fix_sum(h_plus)
                h_minus = h.copy()
                h_minus[i] -= eps
                h_minus = np.clip(h_minus, 0.0, 1.0)
                h_minus = _fix_sum(h_minus)
                grad[i] = (_score(h_plus) - _score(h_minus)) / (2 * eps)
            return grad
        
        h_init = h_best.copy()
        bounds = [(0.0, 1.0)] * N
        constraints = [
            {'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0,
             'jac': lambda h: np.ones(N)}
        ]
        
        try:
            result = minimize(
                _polish_obj,
                h_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            h_candidate = np.clip(result.x, 0.0, 1.0)
            h_candidate = _fix_sum(h_candidate)
            sc = _score(h_candidate)
            if sc < score_best:
                score_best = sc
                h_best = h_candidate.copy()
        except Exception:
            pass
    
    # Final verification
    h_best = np.clip(h_best, 0.0, 1.0)
    h_best = _fix_sum(h_best)
    
    return h_best
