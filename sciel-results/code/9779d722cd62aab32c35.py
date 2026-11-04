"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_overlaps(h):
    """Compute overlap(s) for all shifts s."""
    N = len(h)
    # C(s) = sum_i h[i] * h[(i+s) % N]
    C = np.array([np.dot(h, np.roll(h, s)) for s in range(N)])
    overlaps = N / 2.0 - C
    return overlaps

def _soft_max(overlaps, beta):
    """Soft max of overlaps."""
    m = np.max(overlaps)
    return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta

def _objective_and_grad(h, beta):
    """Objective and gradient for soft-max overlap."""
    overlaps = _compute_overlaps(h)
    val = _soft_max(overlaps, beta)
    
    # Weights
    m = np.max(overlaps)
    w = np.exp(beta * (overlaps - m))
    w /= np.sum(w)
    
    # Gradient: df/dh[j] = sum_s w_s * [(1 - h[(j+s)%N]) - h[(j-s)%N]]
    grad = np.zeros(N)
    for s in range(N):
        if w[s] < 1e-15:
            continue
        # d(overlap(s))/dh[j] = (1 - h[(j+s)%N]) - h[(j-s)%N]
        idx_plus = (np.arange(N) + s) % N
        idx_minus = (np.arange(N) - s) % N
        d_over = (1.0 - h[idx_plus]) - h[idx_minus]
        grad += w[s] * d_over
    
    return val, grad

def _project(h, tol=1e-14):
    """Project h onto {x: 0<=x_i<=1, sum(x)=N/2}."""
    h = np.asarray(h, dtype=float).copy()
    target = N / 2.0
    
    def sum_clipped(t):
        return np.sum(np.clip(h - t, 0.0, 1.0))
    
    lo, hi = -2.0, 2.0
    while sum_clipped(lo) < target:
        lo -= 1.0
    while sum_clipped(hi) > target:
        hi += 1.0
    
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if sum_clipped(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    
    t = (lo + hi) / 2.0
    x = np.clip(h - t, 0.0, 1.0)
    diff = target - np.sum(x)
    if abs(diff) > 1e-15:
        x += diff / N
        x = np.clip(x, 0.0, 1.0)
        diff2 = target - np.sum(x)
        if abs(diff2) > 1e-15:
            x += diff2 / N
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.full(N, 0.5)
    best_score = _soft_max(_compute_overlaps(best_h), beta=100.0)
    
    rng = np.random.default_rng(12345)
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Initialize
        if restart == 0:
            h0 = np.full(N, 0.5)
        elif restart % 2 == 1:
            h0 = rng.uniform(0.1, 0.9, N)
        else:
            # Block patterns
            block_size = rng.choice([2, 3, 4, 5, 7, 10])
            h0 = np.tile(rng.choice([0.2, 0.3, 0.7, 0.8]), N // block_size + 1)[:N]
        
        h0 = _project(h0)
        curr_score = _soft_max(_compute_overlaps(h0), beta=100.0)
        
        # Anneal beta
        betas = [20.0, 50.0, 100.0, 200.0, 500.0]
        
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
            
            def obj(h):
                return _objective_and_grad(h, beta)[0]
            
            def jac(h):
                return _objective_and_grad(h, beta)[1]
            
            bounds = [(0.0, 1.0)] * N
            result = minimize(
                obj, h0, jac=jac, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            h_opt = _project(result.x)
            score = _soft_max(_compute_overlaps(h_opt), beta=beta)
            
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
            
            h0 = h_opt
        
        # Coordinate descent polish
        if time.time() > deadline - 5.0:
            break
        
        for cd_iter in range(100):
            if time.time() > deadline - 3.0:
                break
            improved = False
            for i in range(N):
                for delta in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]:
                    h_trial = best_h.copy()
                    h_trial[i] += delta
                    h_trial = _project(h_trial)
                    s = _soft_max(_compute_overlaps(h_trial), beta=500.0)
                    if s < best_score - 1e-12:
                        best_score = s
                        best_h = h_trial.copy()
                        improved = True
                        break
            if not improved:
                break
    
    best_h = _project(best_h)
    total = np.sum(best_h)
    if abs(total - N/2.0) > 1e-10:
        best_h += (N/2.0 - total) / N
        best_h = np.clip(best_h, 0.0, 1.0)
        total2 = np.sum(best_h)
        if abs(total2 - N/2.0) > 1e-14:
            best_h += (N/2.0 - total2) / N
    
    return best_h
# EVOLVE-BLOCK-END
