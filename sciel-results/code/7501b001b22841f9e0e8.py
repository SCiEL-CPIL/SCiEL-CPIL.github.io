"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def _compute_om_a(a, Om0, w0, wa):
    """Compute Omega_m(a) for CPL dark energy."""
    Omd0 = 1.0 - Om0
    if a <= 0:
        return 1.0
    a3 = a ** 3
    de_term = Omd0 * np.exp(3.0 * (1.0 + w0) * a) if (1.0 + w0) != 0 else Omd0
    total = Om0 / a3 + de_term
    return Om0 / a3 / total

def _compute_gamma(c, a, Om0, w0, wa):
    """Compute gamma from the six-coefficient formula."""
    Om_a = _compute_om_a(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
    return gamma

def _compute_f(c, a, Om0, w0, wa):
    """Compute f = Om(a)^gamma."""
    gamma = _compute_gamma(c, a, Om0, w0, wa)
    Om_a = _compute_om_a(a, Om0, w0, wa)
    return Om_a ** gamma

def _compute_D_normalized(c, a, Om0, w0, wa, n_points=1000):
    """Compute D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)."""
    if a >= 1.0:
        return 1.0
    x_vals = np.linspace(np.log(a), 0.0, n_points)
    a_vals = np.exp(x_vals)
    f_vals = np.array([_compute_f(c, ai, Om0, w0, wa) for ai in a_vals])
    integral = np.trapz(f_vals, x_vals)
    return np.exp(-integral)

def _solve_reference_ode(Om0, w0, wa, a_values):
    """Solve the growth ODE to get reference f and D values at given a_values."""
    # x = ln(a), integrate from x=0 (a=1) downward
    # Riccati: f' + (3 + H'/H) f - 1.5*Om_m + f^2 = 0
    # D' = f*D
    
    # Sort a_values in descending order (from 1 down to smallest)
    sorted_a = np.sort(a_values)[::-1]
    n = len(sorted_a)
    
    # Use fine integration grid
    x_coarse = np.log(sorted_a)
    x_coarse = np.concatenate([[0.0], x_coarse])  # start from 0
    x_coarse = x_coarse[::-1]  # now from 0 down to log(smallest_a)
    
    # Create fine grid
    n_fine = 2000
    x_fine = np.linspace(0.0, x_coarse[-1], n_fine)
    
    # Initialize
    f_val = 0.55  # at a=1, f(1) ≈ 0.55 for LCDM
    D_val = 1.0
    
    # Store results
    f_results = {}
    D_results = {}
    
    # We need to track which fine grid points correspond to which coarse points
    # Use interpolation after integrating
    
    f_array = np.zeros(n_fine)
    D_array = np.zeros(n_fine)
    
    f_array[0] = f_val
    D_array[0] = D_val
    
    for i in range(n_fine - 1):
        x_i = x_fine[i]
        a_i = np.exp(x_i)
        
        Om_ai = _compute_om_a(a_i, Om0, w0, wa)
        Omd_ai = 1.0 - Om_ai
        w_ai = w0 + wa * (1.0 - a_i)
        Hprime_over_H = 1.5 * Om_ai - 1.5 * (1.0 + w_ai) * Omd_ai
        
        dx = x_fine[i+1] - x_fine[i]  # negative
        
        # RK4 for f
        def f_deriv(x, f):
            a_x = np.exp(x)
            Om_x = _compute_om_a(a_x, Om0, w0, wa)
            Omd_x = 1.0 - Om_x
            w_x = w0 + wa * (1.0 - a_x)
            HpH = 1.5 * Om_x - 1.5 * (1.0 + w_x) * Omd_x
            return 1.5 * Om_x - f**2 - (3.0 + HpH) * f
        
        k1 = f_deriv(x_i, f_val)
        k2 = f_deriv(x_i + 0.5*dx, f_val + 0.5*dx*k1)
        k3 = f_deriv(x_i + 0.5*dx, f_val + 0.5*dx*k2)
        k4 = f_deriv(x_i + dx, f_val + dx*k3)
        
        f_new = f_val + (dx/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        
        # D' = f*D, use average f for D update
        f_avg = 0.5 * (f_val + f_new)
        D_new = D_val * np.exp(f_avg * dx)
        
        f_val = f_new
        D_val = D_new
        f_array[i+1] = f_val
        D_array[i+1] = D_val
    
    # Interpolate to get f and D at the requested a_values
    for j, a_j in enumerate(sorted_a):
        x_j = np.log(a_j)
        # Find position in x_fine
        idx = np.searchsorted(x_fine, x_j)
        if idx == 0:
            f_j = f_array[0]
            D_j = D_array[0]
        elif idx >= n_fine:
            f_j = f_array[-1]
            D_j = D_array[-1]
        else:
            # Linear interpolation
            x0, x1 = x_fine[idx-1], x_fine[idx]
            t = (x_j - x0) / (x1 - x0) if x1 != x0 else 0
            f_j = (1-t)*f_array[idx-1] + t*f_array[idx]
            D_j = (1-t)*D_array[idx-1] + t*D_array[idx]
        
        f_results[a_j] = f_j
        D_results[a_j] = D_j
    
    return f_results, D_results

def _precompute_references(models, a_values):
    """Precompute reference f and D values for all models and a values."""
    ref_f = {}
    ref_D = {}
    
    for (Om0, w0, wa) in models:
        f_res, D_res = _solve_reference_ode(Om0, w0, wa, a_values)
        ref_f[(Om0, w0, wa)] = f_res
        ref_D[(Om0, w0, wa)] = D_res
    
    return ref_f, ref_D

def _evaluate_score_fast(c, models, a_values, ref_f, ref_D, use_smooth=False, smooth_beta=50.0):
    """Compute score using precomputed references."""
    all_errs = []
    
    for (Om0, w0, wa) in models:
        for a in a_values:
            if a <= 0 or a > 1.0:
                continue
            
            f_approx = _compute_f(c, a, Om0, w0, wa)
            f_ref = ref_f[(Om0, w0, wa)].get(a, np.nan)
            
            D_approx = _compute_D_normalized(c, a, Om0, w0, wa)
            D_ref = ref_D[(Om0, w0, wa)].get(a, np.nan)
            
            if np.isnan(f_ref) or np.isnan(D_ref):
                continue
            
            err_f = abs(f_approx - f_ref) / abs(f_ref) if abs(f_ref) > 1e-10 else abs(f_approx - f_ref)
            err_D = abs(D_approx - D_ref) / abs(D_ref) if abs(D_ref) > 1e-10 else abs(D_approx - D_ref)
            
            all_errs.append(err_f)
            all_errs.append(err_D)
    
    if not all_errs:
        return 1e6
    
    if use_smooth:
        beta = smooth_beta
        scaled = np.array(all_errs) * beta
        max_scaled = np.max(scaled)
        if max_scaled > 700:
            smooth_max = max_scaled / beta
        else:
            smooth_max = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) / beta
        return 100.0 * smooth_max
    else:
        return 100.0 * np.max(all_errs)

def _gamma_constraint_check(c, models, a_values):
    """Check that gamma is in [0.1, 1.2] at all locations."""
    for (Om0, w0, wa) in models:
        for a in a_values:
            if a <= 0 or a > 1.0:
                continue
            gamma = _compute_gamma(c, a, Om0, w0, wa)
            if gamma < 0.1 or gamma > 1.2:
                return False
    return True

def _penalized_score_fast(c, models, a_values, ref_f, ref_D, use_smooth=False, smooth_beta=50.0):
    """Score with penalty for gamma constraint violation using precomputed refs."""
    base_score = _evaluate_score_fast(c, models, a_values, ref_f, ref_D, use_smooth, smooth_beta)
    if not _gamma_constraint_check(c, models, a_values):
        return base_score + 1000.0
    return base_score

def _optimize_coeffs():
    """Optimize the six coefficients using precomputed references."""
    start_time = time.time()
    deadline = start_time + 80.0
    
    models = [
        (0.3, 0.0, 0.0),
        (0.3, -0.5, 0.5),
        (0.3, -1.2, 0.8),
        (0.25, -0.9, 0.3),
        (0.35, -0.7, -0.2),
        (0.28, -1.0, 0.6),
    ]
    
    a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    bounds = [(-10, 10)] * 6
    
    # Precompute references
    print("Precomputing references...")
    ref_f, ref_D = _precompute_references(models, a_values)
    print(f"References precomputed in {time.time() - start_time:.1f}s")
    
    best_c = None
    best_score = 1e6
    
    try:
        # Phase 1: Differential Evolution
        de_result = differential_evolution(
            _penalized_score_fast,
            bounds,
            args=(models, a_values, ref_f, ref_D, False, 50.0),
            maxiter=100,
            popsize=20,
            tol=1e-8,
            seed=42,
            polish=False,
            init='sobol'
        )
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_c = de_result.x
        print(f"DE done in {time.time() - start_time:.1f}s, score={best_score:.4f}")
        
        # Phase 2: Nelder-Mead polish with smooth surrogate
        if time.time() < deadline - 15.0:
            nm_result = minimize(
                _penalized_score_fast,
                best_c,
                args=(models, a_values, ref_f, ref_D, True, 50.0),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if nm_result.fun < best_score:
                best_score = nm_result.fun
                best_c = nm_result.x
            print(f"NM smooth done in {time.time() - start_time:.1f}s, score={best_score:.4f}")
        
        # Phase 3: High-precision polish with exact objective
        if time.time() < deadline - 5.0:
            polish_result = minimize(
                _penalized_score_fast,
                best_c,
                args=(models, a_values, ref_f, ref_D, False, 50.0),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if polish_result.fun < best_score:
                best_score = polish_result.fun
                best_c = polish_result.x
            print(f"NM exact done in {time.time() - start_time:.1f}s, score={best_score:.4f}")
        
    except Exception as e:
        print(f"Optimization error: {e}")
        best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_c = np.clip(best_c, -10.0, 10.0)
    
    return list(best_c)

# Pre-compute optimized coefficients at import time
_OPTIMIZED_COEFFS = _optimize_coeffs()

def construct():
    return _OPTIMIZED_COEFFS
# EVOLVE-BLOCK-END
