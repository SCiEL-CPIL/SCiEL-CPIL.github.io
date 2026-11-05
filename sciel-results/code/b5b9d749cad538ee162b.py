import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400

def compute_value(h):
    """Compute the task value."""
    h_sum = np.sum(h)
    if h_sum == 0 or not np.isfinite(h_sum):
        return float('inf')
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    return abs(2 * N * max_conv / (h_sum ** 2))

def _build_golay(params):
    """
    Build h from a Golay complementary pair representation.
    
    params = [a0..a_{N-1}, b0..b_{N-1}, offset]
    where a and b are the two complementary sequences, and offset is DC.
    
    h[n] = offset + a[n] + b[n]
    
    The complementary property means:
      autocorr(a) + autocorr(b) = 2*N*delta[k]
    so the off-peak autocorrelation of h comes only from cross terms and offset effects.
    """
    a = params[:N]
    b = params[N:2*N]
    offset = params[2*N]
    h = offset + a + b
    return h

def _golay_objective(params):
    h = _build_golay(params)
    s = np.sum(h)
    if abs(s) < 1e-10 or not np.isfinite(s):
        return 1e6
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    return abs(2.0 * N * max_conv / (s * s))

def _init_golay_pair(rng):
    """
    Initialize a Golay-like complementary pair.
    Use random +/-1 sequences as starting point, then add small random perturbations.
    """
    a = rng.choice([-1.0, 1.0], size=N)
    b = rng.choice([-1.0, 1.0], size=N)
    
    # Add small continuous perturbations to allow gradient-based optimization
    a = a + rng.normal(0, 0.1, N)
    b = b + rng.normal(0, 0.1, N)
    
    offset = rng.uniform(0.5, 2.0)
    params = np.concatenate([a, b, [offset]])
    return params

def _optimize_golay(params_init, deadline, max_de_iter=30):
    """
    Optimize the Golay pair parameters.
    Uses L-BFGS-B with multiple restarts on the full parameter vector.
    """
    best_params = params_init.copy()
    best_val = _golay_objective(best_params)
    
    # L-BFGS-B optimization
    try:
        res = minimize(
            _golay_objective,
            best_params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Nelder-Mead refinement
    if time.time() < deadline - 2:
        try:
            lm_res = minimize(
                _golay_objective, best_params, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if lm_res.fun < best_val:
                best_val = lm_res.fun
                best_params = lm_res.x
        except Exception:
            pass
    
    return best_params, best_val

def _local_refine(h, deadline, max_iter=5000):
    """
    Local refinement: try small random perturbations and keep improvements.
    """
    best_val = compute_value(h)
    best_h = h.copy()
    rng = np.random.default_rng(42)
    
    step_size = 0.05
    for step in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try perturbing 1-5 random indices
        n_perturb = rng.integers(1, 6)
        idx = rng.choice(N, size=n_perturb, replace=False)
        h_new = best_h.copy()
        h_new[idx] += rng.normal(0, step_size, n_perturb)
        
        v_new = compute_value(h_new)
        if np.isfinite(v_new) and v_new < best_val - 1e-10:
            best_val = v_new
            best_h = h_new
            step_size *= 0.97
        elif step % 500 == 0 and step > 0:
            step_size *= 0.995
    
    return best_h, best_val

def _subgradient_polish(h, deadline, max_iter=300):
    """
    Subgradient descent on h directly, normalized to sum=1.
    """
    h = h.copy()
    h_sum = np.sum(h)
    if abs(h_sum) < 1e-10:
        return h, compute_value(h)
    h = h / h_sum  # normalize to sum 1 for stability
    
    best_val = compute_value(h)
    best_h = h.copy()
    
    eta = 0.01
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        conv = np.convolve(h, h)
        k_star = np.argmax(conv)
        
        j = np.arange(N)
        idx1 = k_star - j
        idx2 = j - k_star
        mask1 = (idx1 >= 0) & (idx1 < N)
        mask2 = (idx2 >= 0) & (idx2 < N)
        
        g = np.zeros(N)
        g[mask1] += h[idx1[mask1]]
        g[mask2] += h[idx2[mask2]]
        
        h = h - eta * g
        h = h - (np.sum(h) - 1) / N
        
        eta *= 0.998
        
        val = compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h, best_val

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)  # Should be 2.0
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Golay complementary pair optimization with multiple restarts
    num_golay_restarts = 6
    for i in range(num_golay_restarts):
        if time.time() - start_time > time_limit - 30:
            break
        
        params_init = _init_golay_pair(rng)
        params_opt, val_opt = _optimize_golay(params_init, time_limit - 10)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = _build_golay(params_opt)
    
    # Phase 2: Subgradient polish on best candidate
    if time.time() - start_time < time_limit - 15:
        best_h, best_val = _subgradient_polish(best_h, time_limit - 10, max_iter=300)
    
    # Phase 3: Final local refinement
    if time.time() - start_time < time_limit - 5:
        best_h, best_val = _local_refine(best_h, time_limit - 2, max_iter=8000)
    
    # Final check
    final_val = compute_value(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
    
    return best_h
