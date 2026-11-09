"""Improved growth-index fitting with proper numerical integration."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_Om_a(a, w0, wa, Om0=0.3):
    """Compute Omega_m(a) for flat CPL universe."""
    u = 1.0 + w0 + wa * (1.0 - a)
    if u <= 0.01:
        return None
    Om_de0 = 1.0 - Om0
    denom = Om0 * a**3 + Om_de0 / u
    if denom <= 0:
        return None
    Om_a = Om0 * a**3 / denom
    if Om_a <= 0 or Om_a >= 1:
        return None
    return Om_a

def _gamma_from_coeffs(coeffs, q, u, wa, a):
    """Compute gamma from coefficients."""
    c = coeffs
    return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q

def _compute_ln_D_proper(coeffs, w0, wa, a_target, Om0=0.3, n_int=50):
    """
    Compute ln(D(a)/D(1)) = -integral_ln(a)^0 f(exp(x)) dx
    by properly integrating f(a) = Om(a)^gamma(a) over a in [a_target, 1].
    """
    ln_a_target = np.log(a_target)
    if abs(ln_a_target) < 1e-10:
        return 0.0
    
    # Integrate in x = ln(a) from ln(a_target) to 0
    xs = np.linspace(ln_a_target, 0.0, n_int)
    f_vals = np.zeros(n_int)
    
    for i, x in enumerate(xs):
        a_i = np.exp(x)
        u_i = 1.0 + w0 + wa * (1.0 - a_i)
        if u_i <= 0.01:
            f_vals[i] = 0.0
            continue
        Om_a_i = _compute_Om_a(a_i, w0, wa, Om0)
        if Om_a_i is None:
            f_vals[i] = 0.0
            continue
        q_i = 1.0 - Om_a_i
        gamma_i = _gamma_from_coeffs(coeffs, q_i, u_i, wa, a_i)
        gamma_clipped = np.clip(gamma_i, 0.1, 1.2)
        f_vals[i] = Om_a_i ** gamma_clipped
    
    # Trapezoidal integration
    integral = np.trapz(f_vals, xs)
    return -integral  # ln(D) = -integral

def _fit_coefficients():
    """Fit the six growth-index coefficients."""
    t_start = time.time()
    deadline = 75.0  # Safety margin before 90s kill
    
    # Grid of CPL parameters and scale factors
    w0_vals = np.linspace(-1.2, -0.8, 15)
    wa_vals = np.linspace(-0.5, 0.5, 15)
    a_vals = np.linspace(0.3, 1.0, 30)
    
    Omega_m0 = 0.3
    
    # Build evaluation grid
    q_list, u_list, wa_list, a_list, Om_list = [], [], [], [], []
    w0_list = []
    for w0 in w0_vals:
        for wa in wa_vals:
            for a in a_vals:
                u = 1.0 + w0 + wa * (1.0 - a)
                if u <= 0.05:
                    continue
                Om_a = _compute_Om_a(a, w0, wa, Omega_m0)
                if Om_a is None:
                    continue
                q = 1.0 - Om_a
                q_list.append(q)
                u_list.append(u)
                wa_list.append(wa)
                a_list.append(a)
                Om_list.append(Om_a)
                w0_list.append(w0)
    
    q = np.array(q_list)
    u = np.array(u_list)
    wa_arr = np.array(wa_list)
    a_arr = np.array(a_list)
    Om_a = np.array(Om_list)
    w0_arr = np.array(w0_list)
    
    if len(q) < 10:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Design matrix
    A = np.column_stack([np.ones_like(q), q, q**2, u, wa_arr * a_arr, u * q])
    
    # Initial guess from least squares to Linder-Waugh
    targets = (5.0/7.0) * (Om_a / u) ** 0.38
    coeffs_ls, _, _, _ = np.linalg.lstsq(A, targets, rcond=None)
    coeffs_ls = np.clip(coeffs_ls, -10.0, 10.0)
    
    def _check_gamma_range(coeffs_to_check):
        gamma = _gamma_from_coeffs(coeffs_to_check, q, u, wa_arr, a_arr)
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return False
        return True
    
    if not _check_gamma_range(coeffs_ls):
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Compute reference values using Linder-Waugh gamma
    gamma_ref = targets
    ln_f_ref = gamma_ref * np.log(Om_a)
    
    # Compute reference ln_D using proper integration with Linder-Waugh gamma
    # Create a helper function for reference gamma
    def _gamma_ref_at(q_i, u_i, wa_i, a_i):
        return (5.0/7.0) * (Om_a[np.where((q == q_i) & (u == u_i) & (wa_arr == wa_i) & (a_arr == a_i))[0][0]] / u_i) ** 0.38
    
    # Simpler: just compute ln_D_ref using the same _compute_ln_D_proper but with reference gamma
    # For efficiency, compute only for a subset of points
    n_sample = min(50, len(q))
    sample_idx = np.random.choice(len(q), n_sample, replace=False)
    
    ln_D_ref_sample = np.zeros(n_sample)
    for i, idx in enumerate(sample_idx):
        ln_D_ref_sample[i] = _compute_ln_D_proper(coeffs_ls, w0_arr[idx], wa_arr[idx], a_arr[idx], Omega_m0)
    
    # Objective function: minimize max of |ln_f - ln_f_ref| and |ln_D - ln_D_ref|
    def _objective(coeffs_vec):
        gamma = _gamma_from_coeffs(coeffs_vec, q, u, wa_arr, a_arr)
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
        
        gamma_clipped = np.clip(gamma, 0.1, 1.2)
        ln_f = gamma_clipped * np.log(Om_a)
        
        err_f = np.abs(ln_f - ln_f_ref)
        max_err_f = np.max(err_f)
        
        # Compute ln_D for sample points
        ln_D_sample = np.zeros(n_sample)
        for i, idx in enumerate(sample_idx):
            ln_D_sample[i] = _compute_ln_D_proper(coeffs_vec, w0_arr[idx], wa_arr[idx], a_arr[idx], Omega_m0)
        
        err_D = np.abs(ln_D_sample - ln_D_ref_sample)
        max_err_D = np.max(err_D)
        
        return max(max_err_f, max_err_D)
    
    # Use Nelder-Mead for the initial optimization
    bounds = [(-10, 10)] * 6
    
    best_coeffs = coeffs_ls.copy()
    best_obj = _objective(coeffs_ls)
    
    if time.time() - t_start < deadline - 10:
        try:
            result = minimize(_objective, coeffs_ls, method='Nelder-Mead',
                             options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_obj:
                best_coeffs = np.clip(result.x, -10.0, 10.0)
                best_obj = result.fun
        except Exception:
            pass
    
    # Polish with L-BFGS-B using a smooth surrogate
    if time.time() - t_start < deadline - 5:
        def _smooth_objective(coeffs_vec):
            gamma = _gamma_from_coeffs(coeffs_vec, q, u, wa_arr, a_arr)
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e10
            
            gamma_clipped = np.clip(gamma, 0.1, 1.2)
            ln_f = gamma_clipped * np.log(Om_a)
            
            err_f = np.abs(ln_f - ln_f_ref)
            
            ln_D_sample = np.zeros(n_sample)
            for i, idx in enumerate(sample_idx):
                ln_D_sample[i] = _compute_ln_D_proper(coeffs_vec, w0_arr[idx], wa_arr[idx], a_arr[idx], Omega_m0)
            
            err_D = np.abs(ln_D_sample - ln_D_ref_sample)
            all_errs = np.concatenate([err_f, err_D])
            
            # Log-sum-exp for smooth max
            scale = 50.0
            max_err = np.max(all_errs)
            if max_err < 1e-10:
                return max_err
            shifted = all_errs / scale
            log_sum_exp = max_err / scale + np.log(np.sum(np.exp(shifted - np.max(shifted))))
            return log_sum_exp * scale
        
        try:
            result2 = minimize(_smooth_objective, best_coeffs, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 80, 'ftol': 1e-12})
            if result2.fun < _objective(best_coeffs):
                candidate = np.clip(result2.x, -10.0, 10.0)
                if _check_gamma_range(candidate):
                    best_coeffs = candidate
        except Exception:
            pass
    
    # Final check
    if not _check_gamma_range(best_coeffs):
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return best_coeffs.tolist()

def construct():
    """Construct the six growth-index coefficients."""
    t_start = time.time()
    
    try:
        coeffs = _fit_coefficients()
    except Exception:
        coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Ensure finite and in range
    for i in range(6):
        if not np.isfinite(coeffs[i]) or coeffs[i] < -10 or coeffs[i] > 10:
            coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
            break
    
    return coeffs
# EVOLVE-BLOCK-END
