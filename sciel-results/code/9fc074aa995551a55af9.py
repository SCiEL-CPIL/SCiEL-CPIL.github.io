"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autocorr_sidelobes(x):
    """Compute aperiodic autocorrelation sidelobes for a ±1 or continuous sequence."""
    n = len(x)
    # R(k) = sum_{i=0}^{n-1-k} x[i] * x[i+k] for k = 1..n-1
    sidelobes = np.zeros(n - 1)
    for k in range(1, n):
        sidelobes[k-1] = np.dot(x[:n-k], x[k:])
    return sidelobes

def _objective_continuous(x):
    """Smooth objective: log-sum-exp of |R(k)| for k=1..N-1."""
    sl = _autocorr_sidelobes(x)
    # Use log-sum-exp for smooth max approximation
    # Scale to avoid overflow
    M = np.max(np.abs(sl))
    if M < 1e-12:
        return 0.0
    # log-sum-exp of abs values
    abs_sl = np.abs(sl)
    max_abs = np.max(abs_sl)
    if max_abs < 1e-12:
        return 0.0
    # Smooth max: max_abs + log(sum(exp((abs_sl - max_abs)/temperature))) * temperature
    temperature = 1.0
    shifted = (abs_sl - max_abs) / temperature
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_abs + temperature * log_sum

def _objective_grad(x):
    """Compute objective and gradient."""
    n = len(x)
    sl = _autocorr_sidelobes(x)
    abs_sl = np.abs(sl)
    max_abs = np.max(abs_sl)
    temperature = 1.0
    
    if max_abs < 1e-12:
        obj = 0.0
        grad = np.zeros(n)
        return obj, grad
    
    shifted = (abs_sl - max_abs) / temperature
    shifted = np.clip(shifted, -50, 50)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum = np.log(sum_exp)
    obj = max_abs + temperature * log_sum
    
    # Gradient: d/dx of log_sum = (1/sum_exp) * sum_j exp(shifted_j) * d/dx |sl_j|
    # d/dx |sl_j| = sign(sl_j) * d/dx sl_j
    # sl_j = sum_{i=0}^{n-1-(j+1)} x[i] * x[i+(j+1)]
    # d/dx[i] sl_j = x[i+(j+1)] if i < n-1-(j+1), plus x[i-(j+1)] if i >= j+1
    # Actually, sl_j for lag k=j+1: sl_j = sum_{i=0}^{n-1-k} x[i]*x[i+k]
    # d/dx[m] sl_j = x[m+k] if m < n-k, plus x[m-k] if m >= k
    
    grad = np.zeros(n)
    for j in range(n-1):
        k = j + 1
        w = exp_shifted[j] / sum_exp / temperature
        sign_sl = np.sign(sl[j])
        if sign_sl == 0:
            continue
        # Contribution to gradient from this lag
        # For each i in [0, n-1-k]: d/dx[i] sl_j = x[i+k]
        # For each i in [k, n-1]: d/dx[i] sl_j = x[i-k]
        # Combined: for index m, coefficient is x[m+k] (if m+k < n) + x[m-k] (if m-k >= 0)
        coeff = np.zeros(n)
        # x[m+k] for m = 0..n-1-k => indices 0..n-1-k get x[k..n-1]
        if n - 1 - k >= 0:
            coeff[:n-k] += x[k:]
        # x[m-k] for m = k..n-1 => indices k..n-1 get x[0..n-1-k]
        if n - 1 - k >= 0:
            coeff[k:] += x[:n-k]
        grad += sign_sl * w * coeff
    
    return obj, grad

def _local_search(x, max_iters=5000):
    """Local search: flip individual elements to improve the max sidelobe."""
    x = x.copy()
    n = len(x)
    best_obj = _autocorr_max(x)
    
    for _ in range(max_iters):
        improved = False
        for i in range(n):
            x[i] = -x[i]
            obj = _autocorr_max(x)
            if obj < best_obj:
                best_obj = obj
                improved = True
            else:
                x[i] = -x[i]
        if not improved:
            break
    return x, best_obj

def _autocorr_max(x):
    """Compute max absolute sidelobe of aperiodic autocorrelation."""
    sl = _autocorr_sidelobes(x)
    return np.max(np.abs(sl))

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin before 120s limit
    
    # Start with random ±1 sequence
    np.random.seed(42)
    x_best = np.random.choice([-1, 1], size=N)
    best_score = _autocorr_max(x_best)
    
    # Try multiple random restarts with gradient-based optimization
    for restart in range(20):
        if time.time() > deadline - 30:
            break
        
        # Random initialization
        x0 = np.random.choice([-1.0, 1.0], size=N)
        
        # Run L-BFGS-B on continuous relaxation
        try:
            result = minimize(
                _objective_continuous,
                x0,
                jac=_objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            # Round to ±1
            x_rounded = np.sign(result.x)
            x_rounded[x_rounded == 0] = 1.0
            
            score = _autocorr_max(x_rounded)
            if score < best_score:
                best_score = score
                x_best = x_rounded.copy()
        except Exception:
            continue
        
        # Local search refinement
        if time.time() > deadline - 20:
            break
        x_ls, ls_score = _local_search(x_rounded, max_iters=2000)
        if ls_score < best_score:
            best_score = ls_score
            x_best = x_ls.copy()
    
    # Final local search on best
    if time.time() < deadline - 10:
        x_final, final_score = _local_search(x_best, max_iters=5000)
        if final_score < best_score:
            x_best = x_final
            best_score = final_score
    
    # Ensure output is ±1
    x_best = np.sign(x_best)
    x_best[x_best == 0] = 1.0
    return x_best

# EVOLVE-BLOCK-END
