import numpy as np
from scipy.optimize import minimize
import time

N = 100

def overlap_values(h):
    """Compute overlap(k) for k=1..N-1, linear non-circular."""
    h = np.asarray(h)
    N = len(h)
    overlaps = np.zeros(N-1)
    one_minus_h = 1.0 - h
    for k in range(1, N):
        overlaps[k-1] = np.dot(h[:N-k], one_minus_h[k:])
    return overlaps

def max_overlap(h):
    return np.max(overlap_values(h))

def soft_max_overlap(h, beta=50.0):
    ov = overlap_values(h)
    m = np.max(ov)
    return m + np.log(np.sum(np.exp(beta*(ov - m))))/beta

def construct():
    start = time.time()
    deadline = start + 120.0  # safety margin
    
    # Initial guess: smooth perturbation
    # Use a low-frequency cosine
    t = np.arange(N)
    x = 0.3 * np.cos(2*np.pi*t/N)
    x -= np.mean(x)  # ensure mean 0
    # clip to [-0.5, 0.5]
    x = np.clip(x, -0.5, 0.5)
    # adjust to make sum exactly 0
    x -= np.mean(x)
    h0 = 0.5 + x
    h0 = np.clip(h0, 0.0, 1.0)
    # fix sum to be exactly N/2
    h0 += (N/2 - np.sum(h0))/N
    h0 = np.clip(h0, 0.0, 1.0)
    h0 += (N/2 - np.sum(h0))/N
    
    best_h = h0.copy()
    best_score = max_overlap(best_h)
    
    # Optimization
    def obj(x):
        h = 0.5 + x
        h = np.clip(h, 0.0, 1.0)
        return soft_max_overlap(h)
    
    def jac(x):
        # numerical gradient
        eps = 1e-7
        g = np.zeros_like(x)
        f0 = obj(x)
        for i in range(N):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            g[i] = (obj(xp) - obj(xm))/(2*eps)
        return g
    
    # Use SLSQP with bounds and equality constraint
    bounds = [(-0.5, 0.5)]*N
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x)}]
    
    x0 = best_h - 0.5
    
    result = minimize(obj, x0, method='SLSQP', bounds=bounds, 
                      constraints=constraints, options={'maxiter': 200, 'ftol': 1e-12})
    
    if result.success or result.fun < best_score:
        h_new = 0.5 + result.x
        h_new = np.clip(h_new, 0.0, 1.0)
        h_new += (N/2 - np.sum(h_new))/N
        h_new = np.clip(h_new, 0.0, 1.0)
        h_new += (N/2 - np.sum(h_new))/N
        score_new = max_overlap(h_new)
        if score_new < best_score:
            best_h = h_new
            best_score = score_new
    
    # Check deadline
    if time.time() > deadline:
        return best_h
    
    # Polish: local optimization on the max directly
    # Try to reduce the max overlap further
    for _ in range(50):
        if time.time() > deadline:
            break
        ov = overlap_values(best_h)
        k_max = np.argmax(ov) + 1
        # Try to reduce overlap at k_max by adjusting h
        # Gradient descent on the max overlap
        h = best_h.copy()
        for step in range(100):
            ov = overlap_values(h)
            k_m = np.argmax(ov) + 1
            # Compute gradient of overlap(k_m) w.r.t. h
            grad = np.zeros(N)
            one_minus_h = 1.0 - h
            # d/dh_i [sum_{j=0}^{N-1-k} h_j (1-h_{j+k})]
            # For h_i: appears as h_i in term j=i (if i <= N-1-k) and as (1-h_i) in term j=i-k (if i-k >= 0)
            # Term j=i: h_i (1-h_{i+k}) -> derivative w.r.t h_i: (1-h_{i+k})
            # Term j=i-k: h_{i-k} (1-h_i) -> derivative w.r.t h_i: -h_{i-k}
            if k_m <= N-1:
                # h_i appears in overlap(k_m) for i in 0..N-1-k_m (as h_i) and i in k_m..N-1 (as 1-h_i)
                for i in range(N - k_m):
                    grad[i] += (1.0 - h[i + k_m])
                for i in range(k_m, N):
                    grad[i] -= h[i - k_m]
            # Project gradient to satisfy sum h = N/2: subtract mean
            grad -= np.mean(grad)
            # Clip to [0,1]
            lr = 0.01
            h_new = h - lr * grad
            h_new = np.clip(h_new, 0.0, 1.0)
            h_new += (N/2 - np.sum(h_new))/N
            h_new = np.clip(h_new, 0.0, 1.0)
            h_new += (N/2 - np.sum(h_new))/N
            if max_overlap(h_new) < max_overlap(h):
                h = h_new
            else:
                lr *= 0.5
                if lr < 1e-6:
                    break
        if max_overlap(h) < best_score:
            best_h = h
            best_score = max_overlap(h)
    
    # Final fix: ensure sum is exactly N/2
    best_h += (N/2 - np.sum(best_h))/N
    best_h = np.clip(best_h, 0.0, 1.0)
    # Re-fix sum after clip
    diff = N/2 - np.sum(best_h)
    if abs(diff) > 1e-12:
        # distribute diff
        best_h += diff/N
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
