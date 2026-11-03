import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute C_k(h) for all k.
    C_k(h) = (2/N) * (N/2 - r_k)
    r_k = autocorrelation of h at lag k.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project_simplex(x, target_sum):
    """
    Project x onto the intersection of the box [0, 1]^N and the hyperplane sum(h) = target_sum.
    Uses water-filling algorithm.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    # If sum is already correct and within bounds, return
    if np.all(x >= 0.0) and np.all(x <= 1.0) and abs(np.sum(x) - target_sum) < 1e-12:
        return x

    # Binary search for the offset mu such that sum(clip(x - mu, 0, 1)) = target_sum
    # The function sum(clip(x - mu, 0, 1)) is non-increasing in mu.
    # Lower bound for mu: min(x) - 1 (so all become 1) -> sum = n
    # Upper bound for mu: max(x) (so all become 0) -> sum = 0
    lo = min(x.min() - 1.0, x.min())
    hi = max(x.max() + 1.0, x.max())
    
    # Ensure target is within [0, n]
    if target_sum < 0: target_sum = 0
    if target_sum > n: target_sum = n
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        h_trial = np.clip(x - mid, 0.0, 1.0)
        s = np.sum(h_trial)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
            
    mu = (lo + hi) / 2.0
    h = np.clip(x - mu, 0.0, 1.0)
    
    # Correct for floating point errors
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-14:
        # Distribute error among non-boundary elements
        mask = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            # Fallback: adjust one element
            if diff > 0:
                idx = np.argmin(h)
            else:
                idx = np.argmax(h)
            h[idx] += diff
            h = np.clip(h, 0.0, 1.0)
            
    return h


def _smooth_max_val_grad(h, beta):
    """
    Compute value and gradient of smooth max surrogate.
    f(h) = (1/beta) * log(sum_k exp(beta * C_k(h)))
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta

    # Gradient calculation
    n_fft = 2 * n
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    W = np.fft.rfft(w_pad, n=n_fft)
    
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    
    # sum_k w_k * h_{j+k}
    conv_wh = np.fft.irfft(W * H, n=n_fft)[:n]
    
    # sum_k w_k * h_{j-k}
    h_flip = np.zeros(n_fft)
    h_flip[0] = h[0]
    h_flip[1:n] = h[1:][::-1]
    HF = np.fft.rfft(h_flip, n=n_fft)
    conv_whf = np.fft.irfft(W * HF, n=n_fft)[:n]
    
    grad = -(2.0 / n) * (conv_wh + conv_whf)
    
    return val, grad


def _objective_and_grad(h_flat, beta):
    """
    Objective and gradient for L-BFGS-B.
    We minimize the smooth max surrogate.
    Note: L-BFGS-B handles box constraints, but not the sum constraint.
    We will project the output of L-BFGS-B? No, L-BFGS-B expects unconstrained or box-constrained variables.
    If we use box constraints [0,1], the sum constraint is not enforced during the optimization steps.
    This is a problem.
    
    Alternative: Optimize on the simplex directly? L-BFGS-B doesn't support equality constraints.
    
    Strategy: 
    1. Use L-BFGS-B to minimize the smooth max surrogate with box constraints [0,1].
    2. After each iteration (or at the end), project onto the simplex.
    But L-BFGS-B doesn't allow custom projections in the middle.
    
    Better Strategy:
    Use L-BFGS-B on the smooth max surrogate, but add a large penalty term for the sum constraint violation.
    f(h) = SmoothMax(h) + lambda * (sum(h) - N/2)^2
    Gradient: grad(SmoothMax) + 2*lambda*(sum(h)-N/2)*1
    
    Let's use this penalty approach. It is robust.
    """
    h = h_flat
    val, grad_sm = _smooth_max_val_grad(h, beta)
    
    sum_h = np.sum(h)
    target = N / 2.0
    penalty_val = 1000.0 * (sum_h - target) ** 2
    penalty_grad = 2000.0 * (sum_h - target) * np.ones_like(h)
    
    total_val = val + penalty_val
    total_grad = grad_sm + penalty_grad
    
    return total_val, total_grad


def _solve_lbfgs(h0, beta, max_iter=200):
    """
    Solve using L-BFGS-B with penalty for sum constraint.
    """
    h0 = np.array(h0, dtype=np.float64, copy=True)
    
    res = minimize(
        _objective_and_grad,
        h0,
        args=(beta,),
        method='L-BFGS-B',
        jac=True,
        bounds=[(0.0, 1.0)] * N,
        options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    h_opt = res.x
    # Project to ensure sum constraint is met exactly
    h_opt = _project_simplex(h_opt, N / 2.0)
    
    return h_opt


def _coordinate_descent_polish(h0, max_iter=1000, beta=100.0):
    """
    Coordinate descent polish on the smooth max surrogate.
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project_simplex(h, N / 2.0)
    best_h = h.copy()
    best_score = _score(h)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        improved = False
        # Use random order to avoid bias
        indices = np.random.permutation(N)
        
        for j in indices:
            if time.time() > _DEADLINE - 5:
                break
                
            old_val = h[j]
            # Try to minimize the local contribution
            # We can use a small line search or just try a few steps
            # Let's try steps in [-0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1]
            # But we must maintain sum.
            # Adjusting h[j] by delta requires adjusting others by -delta/(N-1) roughly.
            # This is complex. Instead, let's use the gradient of the smooth max.
            
            # Calculate gradient at current h
            # This is expensive if done for every j.
            # Let's calculate gradient once per outer loop.
            # For now, simple heuristic:
            # If h[j] is large and C_k is large, reduce it?
            
            # Let's stick to a simpler CD: 
            # Pick j, try delta, project, check score.
            for delta in [0.02, -0.02, 0.01, -0.01, 0.05, -0.05]:
                h_new = h.copy()
                h_new[j] = np.clip(old_val + delta, 0.0, 1.0)
                h_new = _project_simplex(h_new, N / 2.0)
                s = _score(h_new)
                if s < best_score - 1e-12:
                    best_score = s
                    best_h = h_new.copy()
                    h = h_new
                    improved = True
                    break
            
        if not improved:
            # If no improvement, try larger steps or break
            pass
            
    return best_h, best_score


def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project_simplex(h, N / 2.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    i = np.arange(N)
    
    # Generate diverse starts
    starts = []
    
    # Random
    for _ in range(10):
        h_rand = np.random.rand(N)
        starts.append(_project_simplex(h_rand, N / 2.0))
        
    # Structured
    for f in [1, 2, 3, 4, 5, 6, 7, 8]:
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            h_sine = 0.5 + 0.3 * np.sin(2 * np.pi * f * i / N + phase)
            starts.append(_project_simplex(h_sine, N / 2.0))
            
    # Mix of frequencies
    for f1 in [1, 2]:
        for f2 in [3, 4, 5]:
            h_mix = 0.5 + 0.2 * np.sin(2 * np.pi * f1 * i / N) + 0.2 * np.cos(2 * np.pi * f2 * i / N)
            starts.append(_project_simplex(h_mix, N / 2.0))
            
    # Initial update
    for h0 in starts:
        _update_best(h0)
        
    # Phase 1: L-BFGS-B refinement with low beta
    if best_h is not None:
        for beta in [10.0, 50.0]:
            if time.time() > _DEADLINE - 30:
                break
            h_candidate = _solve_lbfgs(best_h, beta, max_iter=100)
            _update_best(h_candidate)
            
    # Phase 2: L-BFGS-B with higher beta
    if best_h is not None:
        for beta in [100.0, 200.0]:
            if time.time() > _DEADLINE - 15:
                break
            h_candidate = _solve_lbfgs(best_h, beta, max_iter=200)
            _update_best(h_candidate)
            
    # Phase 3: Coordinate Descent Polish
    if best_h is not None and time.time() < _DEADLINE - 10:
        h_candidate, s_candidate = _coordinate_descent_polish(best_h, max_iter=2000)
        if s_candidate < best_score:
            _update_best(h_candidate)
            
    # Phase 4: Final L-BFGS-B with very high beta
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_candidate = _solve_lbfgs(best_h, beta=500.0, max_iter=300)
        _update_best(h_candidate)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project_simplex(best_h, N / 2.0)
    return best_h
