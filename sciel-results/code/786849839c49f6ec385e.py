import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    timeout = 78.0  # Safety margin: 12s buffer before 90s kill
    
    # Representative cosmologies (Om, w0, wa)
    # We assume the validation set is similar to standard LCDM-like or CPL models
    cosmologies = [
        (0.3, -1.0, 0.0),   # LCDM
        (0.3, -0.9, 0.1),   # CPL
        (0.25, -1.0, 0.0),  # LCDM low Om
        (0.35, -1.0, 0.0),  # LCDM high Om
        (0.3, -0.5, 0.5),   # Quintessence-like
        (0.3, -1.2, 0.5),   # Phantom-like
        (0.3, 0.0, 0.0),    # Matter-like (w=0)
        (0.3, -0.8, 0.2),   # General CPL
        (0.2, -1.0, 0.0),   # Low matter density
        (0.4, -1.0, 0.0),   # High matter density
        (0.3, -0.7, 0.3),   # Another CPL
        (0.3, -1.1, -0.1),  # Slightly phantom
    ]
    
    # Scale factors: denser near a=1 where D is normalized, coarser at small a
    a_vals = np.concatenate([
        np.logspace(-4, -1, 60),   # 60 points from 1e-4 to 0.1
        np.linspace(0.1, 1.0, 140)  # 140 points from 0.1 to 1.0
    ])
    
    # Pre-compute reference solutions for each cosmology
    # D_ref[i] is the normalized growth factor D(a)/D(1)
    # f_ref[i] is the growth rate f(a) = d ln D / d ln a
    ref_data = []
    for i, (om, w0, wa) in enumerate(cosmologies):
        if time.time() - start_time > timeout * 0.4:
            break
        try:
            # Define H^2(a) normalized to H0^2
            # E(a)^2 = Om a^-3 + (1-Om) a^(-3(1+w0+wa))
            
            def E_sq(a, om=om, w0=w0, wa=wa):
                de_term = (1.0 - om) * np.power(a, -3.0 * (1.0 + w0 + wa))
                return om * np.power(a, -3.0) + de_term
            
            def dE_sq_da(a, om=om, w0=w0, wa=wa):
                term_m = -3.0 * om * np.power(a, -4.0)
                term_de = -3.0 * (1.0 + w0 + wa) * (1.0 - om) * np.power(a, -3.0 * (1.0 + w0 + wa) - 1.0)
                return term_m + term_de
            
            def ode(x, y, om=om, w0=w0, wa=wa):
                a = np.exp(x)
                D, Dp = y
                e2 = E_sq(a)
                de2_da = dE_sq_da(a)
                if e2 <= 0:
                    return [0, 0]
                h_prime_over_h = 0.5 * de2_da / e2
                om_a = om * np.power(a, -3.0) / e2
                Dpp = 1.5 * om_a / (a * a) * D - (3.0 + h_prime_over_h) * Dp
                return [Dp, Dpp]
            
            # Integrate from x_min = ln(1e-4) to x_max = 0
            x_min = np.log(1e-4)
            x_max = 0.0
            a0 = np.exp(x_min)
            y0 = [a0, 1.0]
            
            sol = solve_ivp(ode, [x_min, x_max], y0, rtol=1e-10, atol=1e-12, dense_output=True, args=())
            
            if not sol.success:
                continue
                
            # Evaluate on a_vals
            D_arr = sol.sol(np.log(a_vals))[0]
            Dp_arr = sol.sol(np.log(a_vals))[1]
            
            # Normalize D such that D(1) = 1
            D_norm_factor = D_arr[-1]
            D_norm = D_arr / D_norm_factor
            # f = d ln D / d ln a = a D' / D
            f_arr = a_vals * Dp_arr / D_arr
            
            ref_data.append((om, w0, wa, D_norm, f_arr))
        except Exception:
            continue

    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Precompute common quantities for faster objective evaluation
    precomputed = []
    for om, w0, wa, D_ref, f_ref in ref_data:
        e2 = om * np.power(a_vals, -3.0) + (1.0 - om) * np.power(a_vals, -3.0 * (1.0 + w0 + wa))
        om_a = om * np.power(a_vals, -3.0) / e2
        q = 1.0 - om_a
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        x_vals = np.log(a_vals)
        dx = np.diff(x_vals)
        precomputed.append((om_a, q, u, wa, x_vals, dx, D_ref, f_ref))

    def compute_errors(coeffs):
        """Compute all relative errors for f and D across all cosmologies."""
        c0, c1, c2, c3, c4, c5 = coeffs
        
        all_errors = []
        gamma_penalty = 0.0
        
        for om_a, q, u, wa, x_vals, dx, D_ref, f_ref in precomputed:
            gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a_vals + c5 * u * q
            
            # Smooth penalty for gamma constraint violation [0.1, 1.2]
            gamma_lo = np.maximum(0.1 - gamma, 0.0)
            gamma_hi = np.maximum(gamma - 1.2, 0.0)
            gamma_penalty += 1000.0 * (np.max(gamma_lo) + np.max(gamma_hi))
            
            # Clip gamma for safe computation
            gamma_safe = np.clip(gamma, 0.01, 1.19)
            f_app = np.power(om_a, gamma_safe)
            
            # D_app: exp(-integral f dx) from x_i to 0
            # Use Simpson-like weighted trapezoid for better accuracy
            f_avg = (f_app[:-1] + f_app[1:]) / 2.0
            int_steps = f_avg * dx
            int_cum = np.concatenate(([0.0], np.cumsum(int_steps)))
            total_int = int_cum[-1]
            
            exponents = -(total_int - int_cum)
            D_app = np.exp(exponents)
            
            # Relative errors
            err_f = np.abs(f_app - f_ref) / np.maximum(f_ref, 1e-10)
            err_D = np.abs(D_app - D_ref) / np.maximum(D_ref, 1e-10)
            
            all_errors.extend(err_f.tolist())
            all_errors.extend(err_D.tolist())
        
        return np.array(all_errors), gamma_penalty

    def objective(coeffs):
        """Exact max error objective."""
        c0, c1, c2, c3, c4, c5 = coeffs
        coeffs_clipped = np.clip(coeffs, -10, 10)
        errors, gamma_penalty = compute_errors(coeffs_clipped)
        max_err = np.max(errors)
        return max_err + gamma_penalty

    def objective_smooth(coeffs, beta=20.0):
        """Smooth surrogate using log-sum-exp for better gradient behavior."""
        c0, c1, c2, c3, c4, c5 = coeffs
        coeffs_clipped = np.clip(coeffs, -10, 10)
        errors, gamma_penalty = compute_errors(coeffs_clipped)
        
        # Soft max via log-sum-exp
        max_e = np.max(errors)
        # Guard against overflow
        shifted = errors - max_e
        soft_max = max_e + np.log(np.sum(np.exp(shifted))) / beta
        return soft_max + gamma_penalty

    # Initial guess: constant gamma 0.55
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_coeffs = x0.copy()
    best_score = objective(x0)
    
    # Optimization with time-aware strategy
    try:
        bounds = [(-10, 10)] * 6
        elapsed = time.time() - start_time
        
        # Pass 1: L-BFGS-B with smooth objective (fast, gradient-based)
        if elapsed < timeout * 0.5:
            res = minimize(objective_smooth, best_coeffs, method='L-BFGS-B', 
                          bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        
        elapsed = time.time() - start_time
        
        # Pass 2: Nelder-Mead for local refinement (handles non-smoothness)
        if elapsed < timeout * 0.7:
            res2 = minimize(objective, best_coeffs, method='Nelder-Mead', 
                           options={'maxiter': 200, 'fatol': 1e-12, 'xatol': 1e-8})
            if res2.fun < best_score:
                best_score = res2.fun
                best_coeffs = res2.x
        
        elapsed = time.time() - start_time
        
        # Pass 3: High-precision polish with L-BFGS-B on smooth objective
        if elapsed < timeout * 0.85:
            res3 = minimize(objective_smooth, best_coeffs, method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10})
            if res3.fun < best_score:
                best_score = res3.fun
                best_coeffs = res3.x
        
        elapsed = time.time() - start_time
        
        # Pass 4: Final Nelder-Mead polish on exact objective
        if elapsed < timeout * 0.95:
            res4 = minimize(objective, best_coeffs, method='Nelder-Mead', 
                           options={'maxiter': 150, 'fatol': 1e-14, 'xatol': 1e-10})
            if res4.fun < best_score:
                best_score = res4.fun
                best_coeffs = res4.x
                
    except Exception:
        pass
    
    # Final check and clipping
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    # Verify gamma constraint on the final result
    gamma_valid = True
    for om, w0, wa, D_ref, f_ref in ref_data:
        e2 = om * np.power(a_vals, -3.0) + (1.0 - om) * np.power(a_vals, -3.0 * (1.0 + w0 + wa))
        om_a = om * np.power(a_vals, -3.0) / e2
        q = 1.0 - om_a
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        c0, c1, c2, c3, c4, c5 = best_coeffs
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a_vals + c5 * u * q
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            gamma_valid = False
            break
    
    if not gamma_valid:
        # Fallback to safe constant gamma
        best_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    return best_coeffs.tolist()
