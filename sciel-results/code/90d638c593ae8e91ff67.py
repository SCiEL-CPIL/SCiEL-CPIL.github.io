"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_overlap(h, shift):
    """Compute overlap of h with shifted (1-h) for a given shift."""
    # Shifted h: h_shifted[i] = h[(i - shift) % N]
    h_shifted = np.roll(h, shift)
    # 1 - h_shifted
    one_minus_h_shifted = 1.0 - h_shifted
    # Overlap = sum(h * one_minus_h_shifted)
    return np.dot(h, one_minus_h_shifted)


def _max_overlap(h):
    """Compute maximum overlap over all non-zero shifts."""
    max_ov = 0.0
    for s in range(1, N):
        ov = _compute_overlap(h, s)
        if ov > max_ov:
            max_ov = ov
    return max_ov


def _objective_and_grad(x):
    """Objective: sum of squared overlaps for all shifts (smooth surrogate for max).
    This encourages flattening the overlap spectrum."""
    h = x
    overlaps = np.zeros(N - 1)
    for s in range(1, N):
        h_shifted = np.roll(h, s)
        overlaps[s - 1] = np.dot(h, 1.0 - h_shifted)
    
    # Mean overlap
    mean_ov = np.mean(overlaps)
    
    # Objective: sum of squared deviations from mean (variance)
    obj = np.sum((overlaps - mean_ov) ** 2)
    
    # Gradient with respect to h
    # d/dh sum_s (dot(h, 1-h_shifted) - mean)^2
    # Let f_s = dot(h, 1 - roll(h, s))
    # df_s/dh = 1 - roll(h, s) - roll(1 - h, -s)  ... need to be careful
    # Actually f_s(h) = sum_i h[i] * (1 - h[(i-s)%N])
    # df_s/dh[j] = (1 - h[(j-s)%N]) + sum_i h[i] * d/dh[j](1 - h[(i-s)%N])
    # = (1 - h[(j-s)%N]) - sum_i h[i] * delta_{(i-s)%N, j}
    # = (1 - h[(j-s)%N]) - h[(j+s)%N]
    
    grad = np.zeros(N)
    for s in range(1, N):
        idx = s - 1
        dev = overlaps[idx] - mean_ov
        # df_s/dh
        df_s = np.zeros(N)
        for j in range(N):
            df_s[j] = (1.0 - h[(j - s) % N]) - h[(j + s) % N]
        grad += 2.0 * dev * df_s
    
    return obj, grad


def _project(x):
    """Project x onto {h in [0,1], sum(h) = N/2} using bisection on water-filling."""
    target = N / 2.0
    lo, hi = 0.0, 1.0
    
    for _ in range(100):
        mid = (lo + hi) / 2.0
        h_proj = np.clip(x - mid, 0.0, 1.0)
        s = np.sum(h_proj)
        if s > target:
            lo = mid
        else:
            hi = mid
    
    mid = (lo + hi) / 2.0
    h_proj = np.clip(x - mid, 0.0, 1.0)
    return h_proj


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Initialize with small perturbation around 0.5
    np.random.seed(42)
    h = np.full(N, 0.5) + np.random.randn(N) * 0.05
    h = _project(h)
    
    best_h = h.copy()
    best_obj = np.inf
    best_max_ov = _max_overlap(h)
    
    # Use L-BFGS-B with constraints
    # We'll minimize the variance of overlaps as a smooth surrogate
    # Then do a final polish
    
    result = minimize(
        _objective_and_grad,
        h,
        jac=True,
        method='L-BFGS-B',
        bounds=[(0.0, 1.0)] * N,
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if time.time() > deadline - 30:
        # Return best found so far
        return best_h
    
    if result.success or result.fun < best_obj:
        h_new = _project(result.x)
        obj_new = _objective_and_grad(h_new)[0]
        if obj_new < best_obj:
            best_h = h_new.copy()
            best_obj = obj_new
            best_max_ov = _max_overlap(h_new)
    
    # Now do a local search to minimize the actual max overlap
    # Use subgradient-like approach with projection
    
    h = best_h.copy()
    lr = 0.01
    for it in range(500):
        if time.time() > deadline - 10:
            break
        
        # Compute all overlaps
        overlaps = np.zeros(N - 1)
        for s in range(1, N):
            h_shifted = np.roll(h, s)
            overlaps[s - 1] = np.dot(h, 1.0 - h_shifted)
        
        max_ov = np.max(overlaps)
        
        if max_ov < best_max_ov:
            best_max_ov = max_ov
            best_h = h.copy()
        
        # Find shifts that achieve (near) max overlap
        threshold = max_ov - 0.001
        active_shifts = np.where(overlaps >= threshold)[0] + 1
        
        if len(active_shifts) == 0:
            active_shifts = np.array([np.argmax(overlaps) + 1])
        
        # Gradient of max overlap w.r.t. h
        grad = np.zeros(N)
        for s in active_shifts:
            for j in range(N):
                grad[j] += (1.0 - h[(j - s) % N]) - h[(j + s) % N]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad /= grad_norm
        
        # Update
        h = h - lr * grad
        h = _project(h)
        
        # Adaptive learning rate
        if it % 50 == 0:
            lr *= 0.9
    
    # Final polish with very small steps
    h = best_h.copy()
    lr = 0.001
    for it in range(200):
        if time.time() > deadline - 5:
            break
        
        overlaps = np.zeros(N - 1)
        for s in range(1, N):
            h_shifted = np.roll(h, s)
            overlaps[s - 1] = np.dot(h, 1.0 - h_shifted)
        
        max_ov = np.max(overlaps)
        if max_ov < best_max_ov:
            best_max_ov = max_ov
            best_h = h.copy()
        
        threshold = max_ov - 0.0005
        active_shifts = np.where(overlaps >= threshold)[0] + 1
        if len(active_shifts) == 0:
            active_shifts = np.array([np.argmax(overlaps) + 1])
        
        grad = np.zeros(N)
        for s in active_shifts:
            for j in range(N):
                grad[j] += (1.0 - h[(j - s) % N]) - h[(j + s) % N]
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad /= grad_norm
        
        h = h - lr * grad
        h = _project(h)
    
    return best_h
# EVOLVE-BLOCK-END
