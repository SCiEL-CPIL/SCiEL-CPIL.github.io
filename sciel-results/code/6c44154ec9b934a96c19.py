import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 100

def _score(h):
    """Compute the maximum overlap of h with shifted 1-h."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    """Smooth maximum using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _smooth_max_grad(x, alpha=100.0):
    """Gradient of smooth maximum."""
    x_max = np.max(x)
    e = np.exp(alpha * (x - x_max))
    p = e / np.sum(e)
    return p

def _project_sum(h, target_sum=N/2.0):
    """Project h onto {0<=h<=1, sum(h)=target_sum} via bisection on clip offset."""
    h = h.copy()
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h_clipped = np.clip(h + mid, 0.0, 1.0)
        s = np.sum(h_clipped)
        if s > target_sum:
            hi = mid
        else:
            lo = mid
    return np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)

def _logistic_sequence(x0, r, n):
    """Generate n iterations of the logistic map starting from x0 with parameter r."""
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i-1] * (1.0 - x[i-1])
    # Normalize to [0,1] and ensure sum is N/2
    x = np.clip(x, 0.0, 1.0)
    x = _project_sum(x)
    return x

def _polish_fourier(h_init, max_iter=20000):
    """Polish using Fourier basis optimization."""
    h_best = h_init.copy()
    score_best = _score(h_best)
    x = np.arange(N)
    
    for K in [2, 3, 4, 5, 6, 8, 10]:
        n_params = 2 * K
        basis = np.zeros((N, 2*K))
        for k in range(1, K+1):
            basis[:, 2*(k-1)] = np.cos(2 * np.pi * k * x / N)
            basis[:, 2*(k-1)+1] = np.sin(2 * np.pi * k * x / N)
        
        def get_h(params):
            return 0.5 + basis @ params
        
        def project(h):
            return _project_sum(h)
        
        def objective(params):
            h = get_h(params)
            h_c = project(h)
            shifted = np.zeros(N)
            shifted[:-1] = h_c[1:]
            overlap = h_c * (1.0 - shifted)
            return _smooth_max(overlap, alpha=300.0)
        
        def objective_grad(params):
            h = get_h(params)
            h_c = project(h)
            shifted = np.zeros(N)
            shifted[:-1] = h_c[1:]
            overlap = h_c * (1.0 - shifted)
            p = _smooth_max_grad(overlap, alpha=300.0)
            
            grad_hc = np.zeros(N)
            h_next = np.zeros(N)
            h_next[:-1] = h_c[1:]
            grad_hc += p * (1.0 - h_next)
            h_prev = np.zeros(N)
            h_prev[1:] = h_c[:-1]
            p_prev = np.zeros(N)
            p_prev[1:] = p[:-1]
            grad_hc -= p_prev * h_c
            
            mask = (h > 0.01) & (h < 0.99)
            grad_h = grad_hc * mask
            return basis.T @ grad_h
        
        # Try multiple initial points
        for init_seed in [42, 123, 456, 789]:
            np.random.seed(init_seed)
            h0 = np.random.randn(n_params) * 0.01
            
            try:
                res = minimize(
                    objective,
                    h0,
                    jac=objective_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                h_candidate = project(get_h(res.x))
                sc = _score(h_candidate)
                if sc < score_best:
                    score_best = sc
                    h_best = h_candidate.copy()
                    
            except Exception:
                pass
    
    return h_best, score_best

def _polish_direct(h_init, max_iter=10000):
    """Direct coordinate descent polish."""
    h = h_init.copy()
    h = _project_sum(h)
    best_score = _score(h)
    best_h = h.copy()
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            h_old = h[i]
            best_local = h_old
            best_local_score = best_score
            
            for delta in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]:
                h_test = h.copy()
                h_test[i] = np.clip(h_old + delta, 0.0, 1.0)
                h_test = _project_sum(h_test)
                sc = _score(h_test)
                if sc < best_local_score:
                    best_local_score = sc
                    best_local = h_test.copy()
            
            if best_local_score < best_score - 1e-16:
                h = best_local
                best_score = best_local_score
                best_h = h.copy()
                improved = True
        
        if not improved:
            break
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    h_best = np.full(N, 0.5)
    score_best = _score(h_best)
    
    # Phase 1: Explore chaotic logistic map sequences
    # Search over (x0, r) where r is in chaotic regime
    def _logistic_objective(params):
        x0, r = params
        # Ensure r is in valid range
        r = np.clip(r, 3.0, 4.0)
        h = _logistic_sequence(x0, r, N)
        return _score(h)
    
    # Use differential evolution for global search over (x0, r)
    bounds = [(0.01, 0.99), (3.2, 4.0)]
    
    try:
        de_result = differential_evolution(
            _logistic_objective,
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            polish=False,
            update_interval=1,
            workers=1
        )
        
        if de_result.fun < score_best:
            x0_opt, r_opt = de_result.x
            h_candidate = _logistic_sequence(x0_opt, r_opt, N)
            score_candidate = _score(h_candidate)
            if score_candidate < score_best:
                score_best = score_candidate
                h_best = h_candidate.copy()
                
    except Exception:
        pass
    
    # Try several manually chosen chaotic parameters
    chaotic_params = [
        (0.5, 3.9), (0.3, 3.99), (0.7, 3.8), (0.1, 3.95),
        (0.9, 3.7), (0.2, 3.999), (0.4, 3.85), (0.8, 3.92)
    ]
    
    for x0, r in chaotic_params:
        if time.time() > deadline - 30:
            break
        h_candidate = _logistic_sequence(x0, r, N)
        sc = _score(h_candidate)
        if sc < score_best:
            score_best = sc
            h_best = h_candidate.copy()
    
    # Phase 2: Fourier refinement
    if time.time() < deadline - 20:
        h_polished, sc_polished = _polish_fourier(h_best)
        if sc_polished < score_best:
            score_best = sc_polished
            h_best = h_polished
    
    # Phase 3: Direct coordinate descent polish
    if time.time() < deadline - 5:
        h_final, sc_final = _polish_direct(h_best, max_iter=5000)
        if sc_final < score_best:
            score_best = sc_final
            h_best = h_final
    
    return h_best
