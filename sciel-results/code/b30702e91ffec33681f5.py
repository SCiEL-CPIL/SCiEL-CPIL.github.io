import numpy as np
import time
from scipy.optimize import minimize

# ---- Task constants (interface) ----
N = 50
DELTA = 0.5 / N
_XC = -0.25 + (np.arange(N) + 0.5) * DELTA

_DEADLINE = 145.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _ratio(h):
    """Raw objective: L2^2 / (L1 * max_autoconv)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-9):
        return -np.inf
    h = np.clip(h, 0, None)
    l1 = np.sum(h)
    if l1 <= 0.0:
        return 0.0
    l2_sq = np.dot(h, h)
    if l2_sq <= 0.0:
        return 0.0
    conv = np.convolve(h, h, mode="full")
    m = np.max(conv)
    if m <= 0.0:
        return 0.0
    return l2_sq / (l1 * m)

def _smoothed_log_ratio(h, tau=0.01):
    """
    Computes -log(R) with smoothed max for the autoconvolution.
    R = L2^2 / (L1 * C_max).
    Assuming L1=1 (normalized h), R = L2^2 / C_max.
    We minimize -log(R) = log(C_max) - log(L2^2).
    C_max is approximated by tau * log(sum(exp(c_k/tau))).
    """
    h = np.clip(h, 0, None)
    l1 = np.sum(h)
    if l1 <= 1e-12:
        return 1e6, np.zeros(N)
    
    # Normalize to sum 1 to keep L1 fixed in the objective logic
    # But we want the gradient w.r.t unnormalized h? 
    # Easier: Optimize on normalized h. 
    # Constraint sum(h)=1 is handled by SLSQP/L-BFGS-B if we use it, 
    # or we can just project.
    # For SLSQP/L-BFGS-B, we define the function on h, and add constraint sum(h)=1.
    
    l2_sq = np.dot(h, h)
    if l2_sq < 1e-12:
        return 1e6, np.zeros(N)
    
    conv = np.convolve(h, h, mode="full")
    
    # Smoothed max
    # Shift to avoid overflow
    c_max = np.max(conv)
    if c_max == 0:
        return 1e6, np.zeros(N)
    
    # log-sum-exp trick
    shifted = conv / tau
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    c_smooth = tau * log_sum_exp
    
    # Objective: minimize -log(R) = -log(l2_sq) + log(c_smooth)  (since L1=1)
    # Actually R = l2_sq / (l1 * c_smooth).
    # If we enforce l1=1, R = l2_sq / c_smooth.
    # -log(R) = log(c_smooth) - log(l2_sq).
    
    val = np.log(c_smooth) - np.log(l2_sq)
    
    # Gradient
    # d/dh log(c_smooth) = (1/c_smooth) * d(c_smooth)/dh
    # d(c_smooth)/dh_i = sum_k (exp(c_k/tau) / sum(exp(c_j/tau))) * d(c_k)/dh_i
    # d(c_k)/dh_i = 2 * h_{k-i} (with bounds)
    
    weights = np.exp(shifted - max_shifted)
    weights /= np.sum(weights)
    
    grad_c = np.zeros(N)
    # We need to sum weights_k * 2 * h_{k-i} for all k
    # This is a convolution of weights with h, but weights are for k in 0..2N-2
    # c_k = sum_j h_j h_{k-j}
    # d(c_k)/dh_i = 2 h_{k-i}
    
    # Efficiently:
    # grad_c_i = 2 * sum_k weights_k h_{k-i}
    # This is 2 * (weights * h)_{i} ? 
    # Note indices: k-i = j => k = i+j.
    # grad_c_i = 2 * sum_j weights_{i+j} h_j
    
    # We can compute this by padding h and weights
    # Or just loop since N=50 is small
    for i in range(N):
        s = 0.0
        for k in range(2 * N - 1):
            j = k - i
            if 0 <= j < N:
                s += weights[k] * h[j]
        grad_c[i] = 2.0 * s
        
    grad_c_smooth = grad_c / c_smooth
    
    # d/dh -log(l2_sq) = - d/dh log(sum h^2) = - 2h / l2_sq
    grad_l2 = -2.0 * h / l2_sq
    
    grad = grad_c_smooth + grad_l2
    
    return val, grad

def _optimize_direct(h_init, max_iter=1000, ftol=1e-10, tau=0.01):
    """
    Optimize h to maximize ratio using L-BFGS-B on the smoothed log-ratio.
    Constraint: sum(h) = 1, h >= 0.
    """
    h_init = np.asarray(h_init, dtype=np.float64)
    h_init = np.clip(h_init, 0, None)
    s = np.sum(h_init)
    if s <= 0:
        h_init = np.ones(N) / N
        s = 1.0
    h_init = h_init / s
    
    def obj_and_grad(x):
        val, grad = _smoothed_log_ratio(x, tau)
        return val, grad
    
    bounds = [(0.0, None)] * N
    
    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones(N)}
    ]
    
    try:
        res = minimize(
            obj_and_grad,
            h_init,
            method='SLSQP', # SLSQP handles constraints better than L-BFGS-B for equality
            jac=True,
            bounds=bounds,
            constraints=cons,
            options={'maxiter': max_iter, 'ftol': ftol}
        )
        
        h_res = res.x
        h_res = np.clip(h_res, 0, None)
        s = np.sum(h_res)
        if s > 0:
            h_res = h_res / s
        else:
            return None, 0.0
            
        ratio_res = _ratio(h_res)
        return h_res, ratio_res
    except Exception:
        pass
    return None, 0.0

def _find_sidon_support(target_size, N):
    """
    Greedy construction of a Sidon set.
    """
    if target_size <= 0:
        return []
    
    support = [0]
    used_sums = {0}
    
    for _ in range(target_size - 1):
        best_candidate = None
        best_score = -1
        
        for candidate in range(N):
            if candidate in support:
                continue
            
            # Count collisions
            collisions = 0
            for s in support:
                if candidate + s in used_sums:
                    collisions += 1
            
            # We want to minimize collisions
            score = -collisions
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        if best_candidate is not None:
            support.append(best_candidate)
            for s in support:
                used_sums.add(best_candidate + s)
        
        if best_candidate is None:
            break
            
    return support

def construct():
    np.random.seed(12345)
    
    best_h = None
    best_ratio = -np.inf

    # 1. Start with uniform distribution
    h_uniform = np.ones(N) / N
    h_opt, s_opt = _optimize_direct(h_uniform, max_iter=200, ftol=1e-8, tau=0.05)
    if h_opt is not None and s_opt > best_ratio:
        best_ratio = s_opt
        best_h = h_opt.copy()

    # 2. Try Sidon-like structured supports
    # Smaller Sidon sets might allow for "purer" structures
    for target_k in [5, 6, 7, 8, 10, 12]:
        if _time_left() < 15.0:
            break
        
        support = _find_sidon_support(target_k, N)
        if len(support) >= 3:
            h_sidon = np.zeros(N)
            h_sidon[support] = 1.0 / len(support)
            
            # Optimize from this start
            h_opt, s_opt = _optimize_direct(h_sidon, max_iter=200, ftol=1e-8, tau=0.05)
            if h_opt is not None and s_opt > best_ratio:
                best_ratio = s_opt
                best_h = h_opt.copy()

    # 3. Try random sparse starts with varying support sizes
    # This helps escape local optima of the structured starts
    num_restarts = 20
    for i in range(num_restarts):
        if _time_left() < 10.0:
            break
        
        K = np.random.randint(3, 15)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        h_opt, s_opt = _optimize_direct(h_rand, max_iter=100, ftol=1e-6, tau=0.05)
        if h_opt is not None and s_opt > best_ratio:
            best_ratio = s_opt
            best_h = h_opt.copy()

    # 4. Polish the best solution found
    # Use tighter tolerances and smaller tau for higher precision
    if best_h is not None and _time_left() > 20.0:
        h_polish_init = best_h.copy()
        h_polish_init[h_polish_init < 1e-6] = 0.0
        s = np.sum(h_polish_init)
        if s > 0:
            h_polish_init /= s
        
        # Try a few different taus for polishing
        for tau in [0.01, 0.005, 0.001]:
            if _time_left() < 5.0:
                break
            h_polished, s_polished = _optimize_direct(h_polish_init, max_iter=500, ftol=1e-10, tau=tau)
            if h_polished is not None and s_polished > best_ratio:
                best_ratio = s_polished
                best_h = h_polished.copy()
                h_polish_init = h_polished.copy() # Update init for next tau

    # 5. If we have time, try a few more random restarts with higher K
    while _time_left() > 10.0:
        K = np.random.randint(5, 20)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        h_opt, s_opt = _optimize_direct(h_rand, max_iter=50, ftol=1e-5, tau=0.1)
        if h_opt is not None and s_opt > best_ratio:
            best_ratio = s_opt
            best_h = h_opt.copy()

    # Fallback
    if best_h is None:
        best_h = np.ones(N) / N

    # Ensure non-negative and normalized
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
