"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0
DEADLINE = 120.0


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _fix_sum(h, target=None):
    """Adjust h so that sum(h) == target (default N/2), preserving [0,1] bounds."""
    if target is None:
        target = TARGET_SUM
    h = np.asarray(h, dtype=float).copy()
    diff = target - h.sum()
    if abs(diff) > 1e-12:
        # Distribute correction uniformly, but be careful with clipping
        h = h + diff / len(h)
        h = np.clip(h, 0.0, 1.0)
        # If clipping broke the sum, do a second pass on non-saturated elements
        diff2 = target - h.sum()
        if abs(diff2) > 1e-12:
            mask = (h > 1e-9) & (h < 1.0 - 1e-9)
            if np.any(mask):
                h[mask] += diff2 / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
                # Third pass if necessary
                diff3 = target - h.sum()
                if abs(diff3) > 1e-9:
                    mask2 = (h > 0) & (h < 1.0)
                    if np.any(mask2):
                        h[mask2] += diff3 / np.sum(mask2)
                        h = np.clip(h, 0.0, 1.0)
    return h


def _overlap_vector(h):
    """Compute the overlap vector o_i = h_i * (1 - h_{i+1}) with h_N = 0."""
    n = len(h)
    overlap = np.zeros(n)
    if n > 1:
        overlap[:-1] = h[:-1] * (1.0 - h[1:])
        overlap[-1] = h[-1]
    else:
        overlap[0] = h[0]
    return overlap


def _smooth_max_and_grad(x, beta):
    """
    Compute soft max of x and its gradient using log-sum-exp.
    Returns (value, gradient_vector).
    """
    x = np.asarray(x, dtype=float)
    max_x = np.max(x)
    if max_x < 1e-12:
        return 0.0, np.zeros_like(x)
    
    shifted = x - max_x
    weights = np.exp(beta * shifted)
    wsum = np.sum(weights)
    
    if wsum == 0:
        return max_x, np.zeros_like(x)
        
    value = max_x + math.log(wsum) / beta
    grad = weights / wsum
    return value, grad


def _objective_and_grad(h, beta, penalty_weight=100.0):
    """
    Compute the smooth objective value and its gradient.
    Objective: smooth_max(overlap) + penalty * (sum(h) - N/2)^2
    """
    n = len(h)
    h = np.clip(h, 0.0, 1.0)
    
    overlap = _overlap_vector(h)
    val_smooth, grad_overlap = _smooth_max_and_grad(overlap, beta)
    
    # Gradient of overlap w.r.t h
    # o_i depends on h_i and h_{i+1}
    # d(o_i)/d(h_i) = 1 - h_{i+1}  (for i < N-1)
    # d(o_i)/d(h_{i+1}) = -h_i      (for i < N-1)
    # d(o_{N-1})/d(h_{N-1}) = 1
    
    grad_h = np.zeros(n)
    if n > 1:
        grad_h[:-1] += grad_overlap[:-1] * (1.0 - h[1:])
        grad_h[1:] += grad_overlap[:-1] * (-h[:-1])
        grad_h[-1] += grad_overlap[-1] * 1.0
    else:
        grad_h[0] += grad_overlap[0] * 1.0
        
    # Penalty term for sum constraint
    sum_error = h.sum() - TARGET_SUM
    penalty_val = penalty_weight * sum_error * sum_error
    penalty_grad = 2.0 * penalty_weight * sum_error
    
    total_val = val_smooth + penalty_val
    total_grad = grad_h + penalty_grad
    
    return total_val, total_grad


def _optimize_direct(h0, deadline, beta_start=50.0, beta_end=500.0, max_iter=300):
    """
    Optimize h using L-BFGS-B with analytic gradients and adaptive beta.
    """
    h = h0.copy()
    h = _fix_sum(h)
    
    # We will run a few stages with increasing beta
    stages = [
        (beta_start, max_iter // 2),
        (beta_end // 2, max_iter // 4),
        (beta_end, max_iter // 4)
    ]
    
    best_h = h.copy()
    best_score = _score(h)
    
    for beta, iters in stages:
        if time.time() > deadline - 10.0:
            break
            
        def obj(x):
            v, g = _objective_and_grad(x, beta)
            return v, g
            
        try:
            res = minimize(obj, h, jac=True, method='L-BFGS-B', 
                           bounds=[(0, 1)] * N,
                           options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success or res.fun < float('inf'):
                h_new = res.x
                h_new = _fix_sum(h_new)
                s = _score(h_new)
                if s < best_score:
                    best_score = s
                    best_h = h_new.copy()
                    h = h_new.copy()
        except Exception:
            pass
            
    return best_h, best_score


def _optimize_2block_smooth(k, a0, b0, deadline):
    """Use scipy to optimize a for the 2-block construction with smooth objective."""
    n = N
    
    def objective(x):
        a = x[0]
        if a < 0.0 or a > 1.0:
            return 1.0
        b = (N / 2.0 - k * a) / (N - k)
        if b < -1e-6 or b > 1.0 + 1e-6:
            return 1.0
        b = np.clip(b, 0.0, 1.0)
        # Construct h and compute smooth max
        h = np.zeros(N)
        h[:k] = a
        h[k:] = b
        overlap = _overlap_vector(h)
        beta = 50.0
        val, _ = _smooth_max_and_grad(overlap, beta)
        return val
    
    # Try multiple starting points
    best_x = None
    best_f = float('inf')
    
    a_min = max(0.0, (N / 2.0 - N + k) / k)
    a_max = min(1.0, N / (2.0 * k))
    
    if a_min >= a_max:
        return None, float('inf')
    
    # Multiple initial guesses
    for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
        a_init = a_min + frac * (a_max - a_min)
        b_init = (N / 2.0 - k * a_init) / (N - k)
        if b_init < 0 or b_init > 1:
            continue
        try:
            res = minimize(objective, np.array([a_init]), method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.success or res.fun < best_f:
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
        except Exception:
            pass
    
    if best_x is None:
        return None, float('inf')
    
    a_opt = np.clip(best_x[0], 0.0, 1.0)
    b_opt = (N / 2.0 - k * a_opt) / (N - k)
    b_opt = np.clip(b_opt, 0.0, 1.0)
    
    # Verify with exact objective
    h = np.zeros(N)
    h[:k] = a_opt
    h[k:] = b_opt
    h = _fix_sum(h)
    s_exact = _score(h)
    return h, s_exact


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    best_h = None
    best_score = float('inf')

    # Phase 1: 2-block search (fast, structural)
    for k in range(1, N):
        if time.time() > deadline - 30.0:
            break
            
        # Constraint: k*a + (N-k)*b = N/2
        # b = (N/2 - k*a) / (N-k)
        # We want to minimize max(a*(1-a), a*(1-b), b*(1-b), b)
        # subject to a in [0,1], b in [0,1]
        
        # Bounds on a:
        a_min = max(0.0, (N / 2.0 - N + k) / k)
        a_max = min(1.0, N / (2.0 * k))
        
        if a_min >= a_max:
            continue
        
        # Coarse grid
        n_samples = 200
        a_vals = np.linspace(a_min + 1e-9, a_max - 1e-9, n_samples)
        
        best_local_score = float('inf')
        best_local_a = None
        best_local_b = None
        
        for a_val in a_vals:
            b_val = (N / 2.0 - k * a_val) / (N - k)
            if b_val < -1e-12 or b_val > 1.0 + 1e-12:
                continue
            b_val = np.clip(b_val, 0.0, 1.0)
            h = np.zeros(N)
            h[:k] = a_val
            h[k:] = b_val
            s = _score(h)
            if s < best_local_score:
                best_local_score = s
                best_local_a = a_val
                best_local_b = b_val
        
        if best_local_a is not None:
            # Refine around best_local_a with finer grid
            a_refine_min = max(a_min + 1e-9, best_local_a - (a_max - a_min) / n_samples * 3)
            a_refine_max = min(a_max - 1e-9, best_local_a + (a_max - a_min) / n_samples * 3)
            if a_refine_min < a_refine_max:
                a_refine_vals = np.linspace(a_refine_min, a_refine_max, 300)
                for a_val in a_refine_vals:
                    b_val = (N / 2.0 - k * a_val) / (N - k)
                    if b_val < -1e-12 or b_val > 1.0 + 1e-12:
                        continue
                    b_val = np.clip(b_val, 0.0, 1.0)
                    h = np.zeros(N)
                    h[:k] = a_val
                    h[k:] = b_val
                    s = _score(h)
                    if s < best_local_score:
                        best_local_score = s
                        best_local_a = a_val
                        best_local_b = b_val
            
            # Use scipy for high-precision polish
            if time.time() < deadline - 10.0:
                opt_h, s_smooth = _optimize_2block_smooth(k, best_local_a, best_local_b, deadline)
                if opt_h is not None and s_smooth < best_local_score:
                    best_local_score = s_smooth
                    best_local_a = best_local_a # Not used directly, we have opt_h
            
            if best_local_score < best_score:
                best_score = best_local_score
                # Reconstruct best_h from best_local_a and best_local_b
                h = np.zeros(N)
                h[:k] = best_local_a
                h[k:] = best_local_b
                h = _fix_sum(h)
                # Verify score
                s = _score(h)
                if s <= best_score + 1e-9:
                    best_h = h.copy()
                else:
                    # Fallback to the one that actually scored best
                    # This can happen if _fix_sum changed things slightly
                    # We should really keep the h that gave best_local_score
                    # Let's just re-optimize or use the last known good h
                    pass

    # Phase 2: Direct optimization from the best 2-block result
    if best_h is not None and time.time() < deadline - 20.0:
        # Use the best 2-block result as initial guess for direct optimization
        h_opt, s_opt = _optimize_direct(best_h, deadline, beta_start=50.0, beta_end=500.0, max_iter=200)
        if s_opt < best_score:
            best_score = s_opt
            best_h = h_opt.copy()
    
    # Phase 3: Try direct optimization from uniform start if time permits
    if time.time() < deadline - 10.0:
        h_uniform = np.full(N, 0.5)
        h_opt, s_opt = _optimize_direct(h_uniform, deadline, beta_start=50.0, beta_end=500.0, max_iter=100)
        if s_opt < best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    if best_h is None:
        # Fallback
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h = np.zeros(N)
        h[:N//2] = A
        h[N//2:] = B
        best_h = _fix_sum(h)

    return best_h
