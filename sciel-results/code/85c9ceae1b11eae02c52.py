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

def _soft_max(conv, beta=20.0):
    """Compute soft maximum using log-sum-exp for smoothness."""
    shifted = beta * conv
    max_shifted = np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted - max_shifted))) + max_shifted
    return log_sum / beta

def _direct_objective(h):
    """Objective for direct optimization of h."""
    s = np.sum(h)
    if abs(s) < 1e-10 or not np.isfinite(s):
        return 1e6
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    return abs(2.0 * N * max_conv / (s * s))

def _direct_objective_soft(h, beta=20.0):
    """Soft version of the objective for smoother gradient-based optimization."""
    s = np.sum(h)
    if abs(s) < 1e-10 or not np.isfinite(s):
        return 1e6
    conv = np.convolve(h, h)
    max_conv_soft = _soft_max(conv, beta)
    return abs(2.0 * N * max_conv_soft / (s * s))

def _build_chirp(params):
    """
    Build h from chirp parameters.
    params = [A, f, beta, offset]
    h[n] = offset + A * cos(2*pi*(f*n + beta*n^2/N))
    """
    A = params[0]
    f = params[1]
    beta = params[2]
    offset = params[3]
    n = np.arange(N, dtype=np.float64)
    phase = 2.0 * np.pi * (f * n + beta * n * n / N)
    h = offset + A * np.cos(phase)
    return h

def _build_sinusoids(params, n_modes):
    """
    Build h from sinusoid parameters.
    params = [offset, freq1, phase1, amp1, freq2, phase2, amp2, ...]
    """
    offset = params[0]
    n = np.arange(N, dtype=np.float64)
    h = np.full(N, offset, dtype=np.float64)
    for k in range(n_modes):
        freq = params[1 + 3*k]
        phase = params[2 + 3*k]
        amp = params[3 + 3*k]
        h += amp * np.cos(2.0 * np.pi * freq * n / N + phase)
    return h

def _chirp_objective(params):
    h = _build_chirp(params)
    s = np.sum(h)
    if abs(s) < 1e-10 or not np.isfinite(s):
        return 1e6
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    return abs(2.0 * N * max_conv / (s * s))

def _sinusoid_objective(params, n_modes):
    h = _build_sinusoids(params, n_modes)
    s = np.sum(h)
    if abs(s) < 1e-10 or not np.isfinite(s):
        return 1e6
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    return abs(2.0 * N * max_conv / (s * s))

def _optimize_chirp(bounds, x0, time_limit=15.0):
    """Run differential evolution + local refinement for chirp model."""
    start = time.time()
    
    # Differential evolution
    try:
        de_result = differential_evolution(
            _chirp_objective, bounds, seed=42, maxiter=100, tol=1e-6,
            popsize=20, polish=True, init='latinhypercube'
        )
        best_params = de_result.x
        best_val = de_result.fun
    except Exception:
        best_params = x0
        best_val = _chirp_objective(x0)
    
    # Local refinement with Nelder-Mead
    if time.time() - start < time_limit - 2:
        try:
            lm_result = minimize(
                _chirp_objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if lm_result.fun < best_val:
                best_params = lm_result.x
                best_val = lm_result.fun
        except Exception:
            pass
    
    return best_params, best_val

def _optimize_sinusoids(n_modes, init_params, time_limit=10.0):
    """Optimize sinusoid parameters using L-BFGS-B."""
    start = time.time()
    
    try:
        res = minimize(
            _sinusoid_objective,
            init_params,
            args=(n_modes,),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
        )
        best_params = res.x
        best_val = _sinusoid_objective(best_params, n_modes)
    except Exception:
        best_params = init_params
        best_val = _sinusoid_objective(init_params, n_modes)
    
    # Local refinement with Nelder-Mead
    if time.time() - start < time_limit - 2:
        try:
            lm_result = minimize(
                _sinusoid_objective, best_params, args=(n_modes,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if lm_result.fun < best_val:
                best_params = lm_result.x
                best_val = lm_result.fun
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
    no_improve = 0
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
            no_improve = 0
            step_size = min(step_size * 1.05, 0.5)  # Slightly increase on success
        else:
            no_improve += 1
            if no_improve % 100 == 0:
                step_size *= 0.95
            if no_improve > 2000:
                # Restart with larger step
                step_size = 0.1
                no_improve = 0
    
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

def _direct_optimize(h0, deadline, max_iter=1000):
    """Direct optimization of h using L-BFGS-B with soft objective."""
    start = time.time()
    try:
        # First try with soft objective for smoother gradients
        res = minimize(
            _direct_objective_soft, h0, method='L-BFGS-B',
            options={'maxiter': max_iter // 2, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': max_iter}
        )
        best_h = res.x
        best_val = _direct_objective(best_h)
        
        # Refine with exact objective
        if time.time() < deadline - 2:
            res2 = minimize(
                _direct_objective, best_h, method='L-BFGS-B',
                options={'maxiter': max_iter // 2, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': max_iter // 2}
            )
            if _direct_objective(res2.x) < best_val:
                best_h = res2.x
                best_val = _direct_objective(best_h)
        
        return best_h, best_val
    except Exception:
        return h0, _direct_objective(h0)

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)  # Should be 2.0
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Chirp optimization with multiple restarts
    bounds_chirp = [
        (0.1, 2.0),      # A: amplitude
        (0.0, 5.0),      # f: frequency
        (0.0, 10.0),     # beta: chirp rate
        (-0.5, 2.0),     # offset: DC level
    ]
    
    num_chirp_restarts = 4
    for i in range(num_chirp_restarts):
        if time.time() - start_time > time_limit - 50:
            break
        
        x0 = np.array([
            rng.uniform(bounds_chirp[0][0], bounds_chirp[0][1]),
            rng.uniform(bounds_chirp[1][0], bounds_chirp[1][1]),
            rng.uniform(bounds_chirp[2][0], bounds_chirp[2][1]),
            rng.uniform(bounds_chirp[3][0], bounds_chirp[3][1]),
        ])
        
        params_opt, val_opt = _optimize_chirp(bounds_chirp, x0, time_limit=10.0)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = _build_chirp(params_opt)
    
    # Phase 2: Sinusoid optimization with varying numbers of modes
    for n_modes in [1, 2, 3, 4]:
        if time.time() - start_time > time_limit - 35:
            break
        
        n_inits = max(2, 6 - n_modes)
        for init_idx in range(n_inits):
            if time.time() - start_time > time_limit - 30:
                break
            
            # Build initial parameters
            offset = rng.uniform(0.5, 2.0)
            params = [offset]
            for k in range(n_modes):
                freq = rng.uniform(1, 10)
                phase = rng.uniform(0, 2 * np.pi)
                amp = rng.uniform(0.1, 1.5)
                params.extend([freq, phase, amp])
            
            params_opt, val_opt = _optimize_sinusoids(n_modes, params, time_limit=7.0)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = _build_sinusoids(params_opt, n_modes)
    
    # Phase 3: Structured pattern attempts
    if time.time() - start_time < time_limit - 25:
        # Alternating sign patterns
        for trial in range(5):
            if time.time() - start_time > time_limit - 20:
                break
            
            a = rng.uniform(0.5, 2.0)
            b = rng.uniform(0.1, 1.0)
            c = rng.uniform(1, 5)
            phase = rng.uniform(0, 2 * np.pi)
            n = np.arange(N)
            h_test = (-1)**n * (a + b * np.sin(2 * np.pi * c * n / N + phase))
            
            if abs(np.sum(h_test)) < 1e-10:
                continue
            
            h_opt, v_opt = _local_refine(h_test, time_limit - 15, max_iter=2000)
            if np.isfinite(v_opt) and v_opt < best_val:
                best_val = v_opt
                best_h = h_opt
    
    # Phase 4: Direct optimization of h (new)
    if time.time() - start_time < time_limit - 20:
        h_opt, v_opt = _direct_optimize(best_h, time_limit - 15, max_iter=500)
        if np.isfinite(v_opt) and v_opt < best_val:
            best_val = v_opt
            best_h = h_opt
    
    # Phase 5: Subgradient polish on best candidate
    if time.time() - start_time < time_limit - 10:
        best_h, best_val = _subgradient_polish(best_h, time_limit - 5, max_iter=200)
    
    # Phase 6: Final local refinement
    if time.time() - start_time < time_limit - 5:
        best_h, best_val = _local_refine(best_h, time_limit - 2, max_iter=10000)
    
    # Final check
    final_val = compute_value(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
    
    return best_h
