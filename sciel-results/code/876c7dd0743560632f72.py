"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
TARGET_SUM = N / 2.0


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    if n == 0:
        return 0.0
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
        # Distribute correction to non-saturated elements
        mask = (h > 1e-9) & (h < 1.0 - 1e-9)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            # Second pass if clipping changed sum
            diff2 = target - h.sum()
            if abs(diff2) > 1e-12:
                mask2 = (h > 0) & (h < 1.0)
                if np.any(mask2):
                    h[mask2] += diff2 / np.sum(mask2)
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


def _smooth_max(x, beta):
    """Smooth maximum via log-sum-exp."""
    m = np.max(x)
    if m < 1e-300:
        return 0.0, np.zeros_like(x)
    shifted = x - m
    weights = np.exp(beta * shifted)
    wsum = np.sum(weights)
    if wsum == 0:
        return m, np.zeros_like(x)
    value = m + np.log(wsum) / beta
    grad = weights / wsum
    return value, grad


def _objective_smooth(h, beta=100.0, penalty_weight=500.0):
    """
    Smooth objective: soft_max(overlap) + penalty * (sum(h) - N/2)^2
    Returns (value, gradient)
    """
    n = len(h)
    h_clip = np.clip(h, 0.0, 1.0)
    
    overlap = _overlap_vector(h_clip)
    val_smooth, grad_overlap = _smooth_max(overlap, beta)
    
    # Gradient of overlap w.r.t h
    grad_h = np.zeros(n)
    if n > 1:
        # o_i = h_i * (1 - h_{i+1}) for i < n-1
        # d(o_i)/d(h_i) = 1 - h_{i+1}
        # d(o_i)/d(h_{i+1}) = -h_i
        grad_h[:-1] += grad_overlap[:-1] * (1.0 - h_clip[1:])
        grad_h[1:] += grad_overlap[:-1] * (-h_clip[:-1])
        # o_{n-1} = h_{n-1} * (1 - 0) = h_{n-1}
        grad_h[-1] += grad_overlap[-1]
    else:
        grad_h[0] += grad_overlap[0]
    
    # Penalty for sum constraint
    sum_error = h_clip.sum() - TARGET_SUM
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
    current_score = _score(h)
    
    # Stages with increasing beta for sharper max approximation
    stages = [
        (beta_start, max_iter // 2),
        (beta_end // 2, max_iter // 4),
        (beta_end, max_iter // 4)
    ]
    
    for beta, iters in stages:
        if time.time() > deadline - 5.0:
            break
        
        def obj(x):
            val, _ = _objective_smooth(x, beta)
            return val
        
        def grad(x):
            _, g = _objective_smooth(x, beta)
            return g
        
        bounds = [(0, 1)] * N
        try:
            res = minimize(obj, h, jac=grad, method='L-BFGS-B', 
                          bounds=bounds, options={'maxiter': iters, 'ftol': 1e-12})
            h_candidate = res.x
            h_candidate = _fix_sum(h_candidate)
            s = _score(h_candidate)
            if s < current_score:
                h = h_candidate
                current_score = s
        except Exception:
            pass
    
    return h, current_score


def _optimize_2block(k, a0, b0, deadline):
    """
    Optimize a 2-block construction: first k elements = a, rest = b.
    Uses scipy with smooth objective.
    """
    def objective(x):
        a = x[0]
        if a < 0.0 or a > 1.0:
            return 1.0
        b = (TARGET_SUM - k * a) / (N - k)
        if b < -1e-6 or b > 1.0 + 1e-6:
            return 1.0
        b = np.clip(b, 0.0, 1.0)
        
        # Build h and compute smooth overlap
        h = np.zeros(N)
        h[:k] = a
        h[k:] = b
        overlap = _overlap_vector(h)
        val, _ = _smooth_max(overlap, beta=100.0)
        return val
    
    a_min = max(0.0, (TARGET_SUM - N + k) / k)
    a_max = min(1.0, TARGET_SUM / k)
    
    if a_min >= a_max:
        return None, float('inf')
    
    best_x = None
    best_f = float('inf')
    
    # Multiple initial guesses
    for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
        a_init = a_min + frac * (a_max - a_min)
        b_init = (TARGET_SUM - k * a_init) / (N - k)
        if b_init < 0 or b_init > 1:
            continue
        try:
            res = minimize(objective, np.array([a_init]), method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass
    
    if best_x is None:
        return None, float('inf')
    
    a_opt = np.clip(best_x[0], 0.0, 1.0)
    b_opt = (TARGET_SUM - k * a_opt) / (N - k)
    b_opt = np.clip(b_opt, 0.0, 1.0)
    
    h = np.zeros(N)
    h[:k] = a_opt
    h[k:] = b_opt
    h = _fix_sum(h)
    s = _score(h)
    return h, s


def _sigmoid_transition_block(k, a, b, center, width, deadline):
    """
    Try a sigmoidal transition between two block values.
    h[i] = a + (b - a) / (1 + exp(-(i - center) / width))
    """
    def objective(params):
        a_val, b_val, c_val, w_val = params
        if a_val < 0 or a_val > 1 or b_val < 0 or b_val > 1:
            return 1.0
        if w_val < 0.5:
            w_val = 0.5
        
        h = np.zeros(N)
        for i in range(N):
            if i < k:
                h[i] = a_val
            else:
                t = (i - center) / w_val
                if t > 50:
                    h[i] = b_val
                elif t < -50:
                    h[i] = a_val
                else:
                    h[i] = a_val + (b_val - a_val) / (1.0 + np.exp(-t))
        
        h = _fix_sum(h)
        overlap = _overlap_vector(h)
        val, _ = _smooth_max(overlap, beta=100.0)
        return val
    
    # Bounds: a in [0,1], b in [0,1], center in [k-5, N-5], width in [0.5, 20]
    bounds = [(0.0, 1.0), (0.0, 1.0), (max(0, k-5), min(N-1, N-5)), (0.5, 20.0)]
    
    try:
        res = differential_evolution(objective, bounds, maxiter=50, popsize=15,
                                     seed=42, tol=1e-6, polish=True,
                                     time_limit=max(1, int(deadline - time.time()) - 10))
        if res.success:
            a_val, b_val, c_val, w_val = res.x
            h = np.zeros(N)
            for i in range(N):
                if i < k:
                    h[i] = a_val
                else:
                    t = (i - c_val) / w_val
                    if t > 50:
                        h[i] = b_val
                    elif t < -50:
                        h[i] = a_val
                    else:
                        h[i] = a_val + (b_val - a_val) / (1.0 + np.exp(-t))
            h = _fix_sum(h)
            s = _score(h)
            return h, s
    except Exception:
        pass
    
    return None, float('inf')


def construct():
    deadline = time.time() + 110.0  # safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Strategy 1: 2-block constructions for various k
    for k in range(10, N - 10, 2):
        if time.time() > deadline - 10.0:
            break
        
        a_min = max(0.0, (TARGET_SUM - N + k) / k)
        a_max = min(1.0, TARGET_SUM / k)
        
        if a_min >= a_max:
            continue
        
        # Coarse grid search
        n_samples = 200
        a_vals = np.linspace(a_min + 1e-9, a_max - 1e-9, n_samples)
        
        best_local_score = float('inf')
        best_local_a = None
        
        for a_val in a_vals:
            b_val = (TARGET_SUM - k * a_val) / (N - k)
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
        
        if best_local_a is not None:
            # Refine with scipy
            if time.time() < deadline - 15.0:
                h_opt, s_opt = _optimize_2block(k, best_local_a, 
                    (TARGET_SUM - k * best_local_a) / (N - k), deadline)
                if h_opt is not None and s_opt < best_local_score:
                    best_local_score = s_opt
                    best_local_a = h_opt[k - 1] if k > 0 else h_opt[0]
            
            if best_local_score < best_score:
                best_score = best_local_score
                b_opt = (TARGET_SUM - k * best_local_a) / (N - k)
                b_opt = np.clip(b_opt, 0.0, 1.0)
                h = np.zeros(N)
                h[:k] = best_local_a
                h[k:] = b_opt
                h = _fix_sum(h)
                best_h = h.copy()
    
    # Strategy 2: Direct optimization with smooth objective
    if time.time() < deadline - 20.0:
        # Start from uniform
        h0 = np.full(N, 0.5)
        h_opt, s_opt = _optimize_direct(h0, deadline)
        if s_opt < best_score:
            best_score = s_opt
            best_h = h_opt.copy()
        
        # Start from best 2-block
        if best_h is not None:
            h_opt2, s_opt2 = _optimize_direct(best_h, deadline)
            if s_opt2 < best_score:
                best_score = s_opt2
                best_h = h_opt2.copy()
    
    # Strategy 3: Sigmoidal transition
    if time.time() < deadline - 10.0:
        for k in [25, 50, 75]:
            if time.time() > deadline - 5.0:
                break
            h_sig, s_sig = _sigmoid_transition_block(k, 0.3, 0.7, k, 5.0, deadline)
            if h_sig is not None and s_sig < best_score:
                best_score = s_sig
                best_h = h_sig.copy()
    
    if best_h is None:
        # Fallback: golden ratio split
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h = np.zeros(N)
        h[:N//2] = A
        h[N//2:] = B
        best_h = _fix_sum(h)
    
    return best_h
