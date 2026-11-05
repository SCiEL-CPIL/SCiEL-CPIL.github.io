"""Improved growth-index ansatz with direct optimization from task structure."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, cumulative_trapezoid
import time
import sys

# EVOLVE-BLOCK-START

# Deadline guard: evaluator kills after 90s; use 80s safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # 55s limit to be safe, leaving time for import/setup

def _check_deadline():
    return time.time() > _DEADLINE

def construct():
    """Return six coefficients [c0..c5] for the growth-index ansatz.
    
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    f = Om(a)^gamma
    D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
    
    We optimize these coefficients to minimize worst-case relative error
    across validation cosmologies for both f and D.
    """
    
    # Initial best result: physically-motivated seed
    # c0=0.55 (standard growth index), c3=-0.5 to cancel leading w0 term
    # This forces remaining coefficients to model subdominant corrections
    best_coeff = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    best_score = np.inf
    
    # Representative validation cosmologies (Om0, w0, wa)
    # We choose a spread to cover the likely test space
    val_cosmos = [
        (0.30, -1.0, 0.0),   # Lambda CDM
        (0.30, -1.0, 0.1),   # Positive wa
        (0.30, -1.0, -0.1),  # Negative wa
        (0.30, -0.9, 0.0),   # w0 > -1
        (0.30, -1.1, 0.0),   # w0 < -1
        (0.28, -1.0, 0.0),   # Lower Om0
        (0.32, -1.0, 0.0),   # Higher Om0
        (0.30, -0.95, 0.05), # Intermediate w0, small positive wa
        (0.30, -1.05, -0.05),# Intermediate w0, small negative wa
        (0.29, -1.0, 0.15),  # Lower Om0, larger wa
    ]
    
    # Scale factor grid for evaluation (log-spaced, denser for better accuracy)
    a_grid = np.logspace(np.log10(0.05), 0.0, 150)
    
    # Pre-compute reference values for each cosmology
    ref_data = []
    
    def _compute_Om(a_vals, Om0, w0, wa):
        """Vectorized computation of Omega_m(a)."""
        a_vals = np.asarray(a_vals)
        n = len(a_vals)
        Om_a = np.ones(n)
        
        valid = a_vals > 0
        if not np.any(valid):
            return Om_a
        
        a_valid = a_vals[valid]
        
        if abs(wa) < 1e-10:
            if abs(1.0 + w0) < 1e-10:
                rho_de_ratio = np.ones_like(a_valid)
            else:
                rho_de_ratio = a_valid ** (-3.0 * (1.0 + w0))
        else:
            exponent = -3.0 * np.log(a_valid) * (1.0 + w0 + wa) + 3.0 * wa * (a_valid - 1.0)
            rho_de_ratio = np.exp(exponent)
        
        rho_m_ratio = a_valid ** -3.0
        denom = Om0 * rho_m_ratio + (1.0 - Om0) * rho_de_ratio
        denom_safe = np.where(denom == 0, 1e-30, denom)
        Om_a[valid] = Om0 * rho_m_ratio / denom_safe
        
        return Om_a
    
    def _compute_reference(Om0, w0, wa, a_vals):
        """
        Compute reference f and D for a given cosmology and a-grid.
        Uses numerical integration of the growth ODE.
        """
        a_vals = np.asarray(a_vals)
        n = len(a_vals)
        
        # 1. Compute Omega_m(a) for all a (vectorized)
        Om_a = _compute_Om(a_vals, Om0, w0, wa)
        
        # 2. Solve ODE for D(a)
        x_grid = np.log(a_vals)
        
        def ode_func(x, y):
            a = np.exp(x)
            # Use vectorized Om computation for single point
            Om = _compute_Om(np.array([a]), Om0, w0, wa)[0]
            
            P = 3.0 + 1.5 * Om
            Q = 1.5 * Om
            return [y[1], -P * y[1] + Q * y[0]]
        
        x_min = np.log(0.1)
        y0 = [0.1, 1.0]
        
        try:
            sol = solve_ivp(ode_func, [x_min, 0.0], y0, t_eval=x_grid, method='RK45', rtol=1e-9, atol=1e-11)
            if not sol.success:
                return None
            D_vals = sol.y[0]
            D_prime_vals = sol.y[1]
            
            idx_1 = np.argmin(np.abs(x_grid))
            D_norm = D_vals[idx_1]
            if D_norm == 0:
                return None
            D_vals = D_vals / D_norm
            D_prime_vals = D_prime_vals / D_norm
            
            f_ref = D_prime_vals
            D_ref = D_vals
            
            return {
                'a': a_vals,
                'Om': Om_a,
                'f': f_ref,
                'D': D_ref,
                'w0': w0,
                'wa': wa
            }
        except Exception:
            return None
    
    for c in val_cosmos:
        ref = _compute_reference(c[0], c[1], c[2], a_grid)
        if ref:
            ref_data.append(ref)
        if _check_deadline():
            break

    if not ref_data:
        return best_coeff.tolist()

    # Pre-compute arrays for fast evaluation
    a_arr = np.array([r['a'] for r in ref_data]).flatten()
    Om_arr = np.array([r['Om'] for r in ref_data]).flatten()
    f_ref_arr = np.array([r['f'] for r in ref_data]).flatten()
    D_ref_arr = np.array([r['D'] for r in ref_data]).flatten()
    w0_arr = np.array([r['w0'] for r in ref_data]).flatten()
    wa_arr = np.array([r['wa'] for r in ref_data]).flatten()
    x_arr = np.log(a_arr)
    
    n_cosmo = len(ref_data)
    n_pts = len(a_grid)
    x_reshaped = x_arr.reshape(n_cosmo, n_pts)
    
    # Pre-compute cumulative trapezoid structure for vectorized D computation
    # For each cosmology, we need: integral from x[i] to x[n-1]
    # = cumulative_trapezoid(f, x, initial=0)[-1] - cumulative_trapezoid(f, x, initial=0)[i]
    
    # Pre-compute the trapezoid weights
    dx = np.diff(x_reshaped, axis=1)  # shape (n_cosmo, n_pts-1)
    dx_half = 0.5 * dx  # shape (n_cosmo, n_pts-1)
    
    # Build the cumulative integral matrix
    # I[i] = sum_{j=1}^{i} dx_half[j-1] * (f[j-1] + f[j])
    # This can be expressed as a matrix multiplication
    # Let's pre-compute the matrix M such that I = M @ f
    # M[i, j] for i >= 1:
    #   M[i, i] += dx_half[i-1]
    #   M[i, i-1] += dx_half[i-1]
    # For efficiency, we'll compute this on the fly using cumsum
    
    def _compute_D_from_f(f_ours_flat):
        """Compute D from f using vectorized cumulative trapezoid.
        f_ours_flat: flat array of length n_cosmo * n_pts
        Returns: flat array of D values
        """
        f_reshaped = f_ours_flat.reshape(n_cosmo, n_pts)
        
        # Compute cumulative trapezoid integrals vectorized
        # I[i] = sum_{j=1}^{i} 0.5*(f[j-1]+f[j])*(x[j]-x[j-1])
        # = sum_{j=1}^{i} dx_half[j-1]*f[j-1] + dx_half[j-1]*f[j]
        
        # For each cosmology, compute the incremental contributions
        # term[j] = dx_half[j-1] * (f[j-1] + f[j]) for j=1..n_pts-1
        terms = dx_half * (f_reshaped[:, :-1] + f_reshaped[:, 1:])  # shape (n_cosmo, n_pts-1)
        
        # Cumulative sum gives I[i] for i=1..n_pts-1
        # I[0] = 0
        cum_I = np.zeros((n_cosmo, n_pts))
        cum_I[:, 1:] = np.cumsum(terms, axis=1)
        
        # We want integral from x[i] to x[n-1] = total - I[i]
        total_I = cum_I[:, -1:]  # shape (n_cosmo, 1)
        I_from_xi_to_end = total_I - cum_I  # shape (n_cosmo, n_pts)
        
        D = np.exp(-I_from_xi_to_end)
        
        return D.flatten()

    def objective(coeff):
        c0, c1, c2, c3, c4, c5 = coeff
        
        q = 1.0 - Om_arr
        u = 1.0 + w0_arr + wa_arr * (1.0 - a_arr)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa_arr*a_arr + c5*u*q
        
        # Check bounds - use soft penalty for smoothness
        gamma_min = np.min(gamma)
        gamma_max = np.max(gamma)
        if gamma_min < 0.1 or gamma_max > 1.2:
            # Soft penalty
            pen = 0.0
            if gamma_min < 0.1:
                pen += 100.0 * (0.1 - gamma_min)**2
            if gamma_max > 1.2:
                pen += 100.0 * (gamma_max - 1.2)**2
            return pen + 1.0
        
        f_ours = np.power(np.maximum(Om_arr, 1e-15), gamma)
        D_ours = _compute_D_from_f(f_ours)
        
        # Relative errors
        mask_f = f_ref_arr > 1e-10
        mask_D = D_ref_arr > 1e-10
        
        err_f = np.zeros_like(f_ours)
        err_D = np.zeros_like(D_ours)
        
        if np.any(mask_f):
            err_f[mask_f] = np.abs((f_ours[mask_f] - f_ref_arr[mask_f]) / f_ref_arr[mask_f])
        
        if np.any(mask_D):
            err_D[mask_D] = np.abs((D_ours[mask_D] - D_ref_arr[mask_D]) / D_ref_arr[mask_D])
        
        # Use smooth max (log-sum-exp) for better gradient behavior
        all_errs = np.concatenate([err_f, err_D])
        all_errs = np.maximum(all_errs, 1e-15)
        
        # Proper soft max: (1/beta) * log(sum(exp(beta * err)))
        # For large beta, this approaches max(err)
        beta = 100.0
        scaled = beta * all_errs
        max_scaled = np.max(scaled)
        soft_max = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) / beta
        
        return soft_max

    # Optimization
    x0 = best_coeff
    
    # Phase 1: Nelder-Mead for global exploration
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        if res.fun < best_score:
            best_coeff = res.x
            best_score = res.fun
    except Exception:
        pass
    
    if _check_deadline():
        return best_coeff.tolist()
    
    # Phase 2: L-BFGS-B with smooth objective for refinement
    try:
        res2 = minimize(objective, best_coeff, method='L-BFGS-B', 
                       bounds=[(-10,10)]*6, 
                       options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
        if res2.fun < best_score:
            best_coeff = res2.x
            best_score = res2.fun
    except Exception:
        pass
    
    if _check_deadline():
        return best_coeff.tolist()
    
    # Phase 3: Final polish with tighter tolerances
    try:
        res3 = minimize(objective, best_coeff, method='L-BFGS-B', 
                       bounds=[(-10,10)]*6, 
                       options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12})
        if res3.fun < best_score:
            best_coeff = res3.x
            best_score = res3.fun
    except Exception:
        pass
    
    # Phase 4: One more refinement pass
    if not _check_deadline():
        try:
            res4 = minimize(objective, best_coeff, method='L-BFGS-B', 
                           bounds=[(-10,10)]*6, 
                           options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-12})
            if res4.fun < best_score:
                best_coeff = res4.x
                best_score = res4.fun
        except Exception:
            pass

    best_coeff = np.clip(best_coeff, -10.0, 10.0)
    
    return best_coeff.tolist()

# EVOLVE-BLOCK-END
