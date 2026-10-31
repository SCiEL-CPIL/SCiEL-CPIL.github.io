import numpy as np
import time
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0

def _score(h):
    """Compute the maximum overlap of h with shifted (1-h)."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _fix_sum(h):
    """Project h to have sum exactly N/2 while staying in [0,1]."""
    h = h.copy()
    for _ in range(100):
        diff = TARGET_SUM - np.sum(h)
        if abs(diff) < 1e-15:
            break
        if diff > 0:
            # Need to increase sum
            idx = np.where(h < 1.0)[0]
            if len(idx) == 0:
                break
            slack = 1.0 - h[idx]
            total_slack = np.sum(slack)
            if total_slack < 1e-15:
                break
            scale = min(1.0, diff / total_slack)
            h[idx] += slack * scale
        else:
            # Need to decrease sum
            idx = np.where(h > 0.0)[0]
            if len(idx) == 0:
                break
            slack = h[idx]
            total_slack = np.sum(slack)
            if total_slack < 1e-15:
                break
            scale = min(1.0, -diff / total_slack)
            h[idx] -= slack * scale
    # Final clamp
    h = np.clip(h, 0.0, 1.0)
    return h

def _overlap_values(h):
    """Compute overlap vector h_i * (1 - h_{i+1}) with boundary h_n = 0."""
    n = len(h)
    h_next = np.zeros(n)
    h_next[:-1] = h[1:]
    return h * (1.0 - h_next)

def _smooth_max(x, alpha=200.0):
    """Smooth approximation of max(x)."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _smooth_max_grad(x, alpha=200.0):
    """Gradient of smooth max w.r.t. x."""
    x_max = np.max(x)
    e = np.exp(alpha * (x - x_max))
    return e / np.sum(e)

def _objective_and_grad(h, alpha=200.0):
    """Compute smooth max of overlap and its gradient w.r.t. h."""
    n = len(h)
    overlap = _overlap_values(h)
    
    # Gradient of overlap_i = h_i * (1 - h_{i+1}) w.r.t. h
    # d(overlap_i)/dh_i = 1 - h_{i+1}
    # d(overlap_i)/dh_{i+1} = -h_i
    h_next = np.zeros(n)
    h_next[:-1] = h[1:]
    h_prev = np.zeros(n)
    h_prev[1:] = h[:-1]
    
    grad_overlap = np.zeros(n)
    grad_overlap += (1.0 - h_next)   # contribution from overlap_i
    grad_overlap -= h_prev           # contribution from overlap_{i-1}
    
    # Smooth max gradient
    p = _smooth_max_grad(overlap, alpha)
    obj = _smooth_max(overlap, alpha)
    grad = p * grad_overlap
    
    return obj, grad

def _local_optimize(h_init, alpha=200.0, max_iter=2000, lr=0.05):
    """Projected gradient descent with line search."""
    h = h_init.copy()
    h = np.clip(h, 0.0, 1.0)
    h = _fix_sum(h)
    
    best_score = _score(h)
    best_h = h.copy()
    
    for it in range(max_iter):
        h_clipped = np.clip(h, 0.0, 1.0)
        h_clipped = _fix_sum(h_clipped)
        
        obj, grad = _objective_and_grad(h_clipped, alpha)
        
        # Gradient step
        h_new = h_clipped - lr * grad
        h_new = np.clip(h_new, 0.0, 1.0)
        h_new = _fix_sum(h_new)
        
        sc_new = _score(h_new)
        
        if sc_new < best_score - 1e-15:
            best_score = sc_new
            best_h = h_new.copy()
            h = h_new.copy()
            lr = min(lr * 1.01, 0.1)
        else:
            lr *= 0.8
            if lr < 1e-10:
                break
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    h_best = np.full(N, 0.5)
    score_best = _score(h_best)
    
    # Strategy 1: Multi-start with diverse initializations
    initializations = []
    
    # Uniform
    initializations.append(np.full(N, 0.5))
    
    # Random
    rng = np.random.RandomState(42)
    for _ in range(5):
        h_rand = rng.uniform(0, 1, N)
        initializations.append(h_rand)
    
    # Sawtooth patterns
    for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for period in [4, 6, 8, 10, 12, 15, 20]:
            h_saw = np.zeros(N)
            h_saw[:period] = a
            h_saw[period:] = 1.0 - a
            # Tile
            n_tiles = (N + period - 1) // period
            h_saw = np.tile(h_saw[:period], n_tiles)[:N]
            initializations.append(h_saw)
    
    # Step function: first k are 1, rest are 0, adjusted for sum
    for k in range(10, 91, 10):
        h_step = np.zeros(N)
        h_step[:k] = 1.0
        initializations.append(h_step)
    
    # Optimize from each initialization
    for h_init in initializations:
        if time.time() > deadline - 30:
            break
        
        h_opt, sc_opt = _local_optimize(h_init, alpha=200.0, max_iter=500, lr=0.05)
        if sc_opt < score_best:
            score_best = sc_opt
            h_best = h_opt.copy()
    
    # Strategy 2: Refine with multiple alpha values
    for alpha in [100.0, 200.0, 400.0, 800.0]:
        if time.time() > deadline - 20:
            break
        
        h_opt, sc_opt = _local_optimize(h_best, alpha=alpha, max_iter=1000, lr=0.03)
        if sc_opt < score_best:
            score_best = sc_opt
            h_best = h_opt.copy()
    
    # Strategy 3: Fourier-based refinement
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
    
    # Final refinement with high alpha
    if time.time() < deadline - 10:
        h_opt, sc_opt = _local_optimize(h_best, alpha=1000.0, max_iter=2000, lr=0.01)
        if sc_opt < score_best:
            score_best = sc_opt
            h_best = h_opt.copy()
    
    # Final verification
    h_best = np.clip(h_best, 0.0, 1.0)
    h_best = _fix_sum(h_best)
    
    return h_best
