"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    # Representative cosmologies for validation
    # Om0, w0, wa
    cosmologies = [
        (0.30, -1.0, 0.0),  # LambdaCDM
        (0.30, -1.0, -0.5), # CPL wa=-0.5
        (0.30, -1.0, 0.5),  # CPL wa=0.5
        (0.25, -1.0, 0.0),  # Lower Om
        (0.35, -1.0, 0.0),  # Higher Om
        (0.30, -0.5, 0.0),  # w0=-0.5
        (0.30, -1.5, 0.0),  # w0=-1.5
    ]
    
    # Use a moderate number of points to keep evaluation fast
    a_values = np.linspace(0.1, 1.0, 50)
    x_values = np.log(a_values)
    
    # Pre-compute reference data for each cosmology
    # ref_data[i] = (f_ref_array, D_ref_array)
    ref_data = []
    
    for Om0, w0, wa in cosmologies:
        Om_DE0 = 1.0 - Om0
        
        def Om_of_a(a):
            # Om_DE(a) = Om_DE0 * a^(-3(1+w0+wa)) * exp(3wa(a-1))
            Om_DE = Om_DE0 * a**(-3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (a - 1.0))
            Om_m = Om0 * a**3
            return Om_m / (Om_m + Om_DE)
        
        def H_prime_over_H(a):
            Om_m = Om0 * a**3
            Om_DE = Om_DE0 * a**(-3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (a - 1.0))
            H2 = Om_m + Om_DE
            
            dOm_m_da = 3.0 * Om_m / a
            dOm_DE_da = Om_DE * (-3.0 * (1.0 + w0 + wa) / a + 3.0 * wa)
            
            dH2_da = dOm_m_da + dOm_DE_da
            
            return 0.5 * a * dH2_da / H2

        # Solve ODE: D'' + (3 + H'/H) D' - 1.5 Om_m D = 0
        def ode_func(x, y):
            a = np.exp(x)
            Om = Om_of_a(a)
            HpH = H_prime_over_H(a)
            return [y[1], -(3.0 + HpH) * y[1] + 1.5 * Om * y[0]]
        
        y0 = [1.0, 1.0]
        
        try:
            sol = solve_ivp(ode_func, [0.0, np.log(0.1)], y0, method='RK45', rtol=1e-9, atol=1e-11, dense_output=True)
            if not sol.success:
                continue
            
            f_ref = np.zeros_like(a_values)
            D_ref = np.zeros_like(a_values)
            
            for i, x in enumerate(x_values):
                y = sol.sol(x)
                D_ref[i] = y[0]
                f_ref[i] = y[1]
            
            ref_data.append((f_ref, D_ref))
        except Exception:
            continue

    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        
        max_rel_err = 0.0
        
        for i, (Om0, w0, wa) in enumerate(cosmologies):
            if i >= len(ref_data): continue
            f_ref, D_ref = ref_data[i]
            
            Om_DE0 = 1.0 - Om0
            
            # Calculate Om(a) for all a_values
            Om_vals = np.zeros_like(a_values)
            for j, a in enumerate(a_values):
                Om_DE = Om_DE0 * a**(-3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (a - 1.0))
                Om_m = Om0 * a**3
                Om_vals[j] = Om_m / (Om_m + Om_DE)
            
            q = 1.0 - Om_vals
            u = 1.0 + w0 + wa * (1.0 - a_values)
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_values + c5*u*q
            
            # Check gamma bounds
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 10.0 + np.max(np.abs(gamma - 0.5))
            
            f_approx = Om_vals**gamma
            
            # Calculate D_approx
            D_approx = np.zeros_like(a_values)
            D_approx[0] = 1.0
            integral = 0.0
            for j in range(1, len(a_values)):
                avg_f = 0.5 * (f_approx[j-1] + f_approx[j])
                dx = x_values[j] - x_values[j-1]
                integral += avg_f * dx
                D_approx[j] = np.exp(integral)
            
            # Relative errors
            mask_f = np.abs(f_ref) > 1e-8
            if np.any(mask_f):
                rel_err_f = np.abs(f_approx[mask_f] - f_ref[mask_f]) / np.abs(f_ref[mask_f])
                max_rel_err = max(max_rel_err, np.max(rel_err_f))
            
            mask_D = np.abs(D_ref) > 1e-8
            if np.any(mask_D):
                rel_err_D = np.abs(D_approx[mask_D] - D_ref[mask_D]) / np.abs(D_ref[mask_D])
                max_rel_err = max(max_rel_err, np.max(rel_err_D))
        
        return max_rel_err

    # Initial guess
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    best_coeffs = x0
    best_err = objective(x0)
    
    # Optimize
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.fun < best_err:
            best_coeffs = res.x
            best_err = res.fun
            
        if time.time() > deadline - 5.0:
            return best_coeffs
            
        # Try a few other starting points if time permits
        start_points = [
            [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.6, -0.1, 0.0, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
            [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],
            [0.55, 0.05, 0.05, 0.05, 0.0, 0.0],
            [0.55, -0.05, 0.05, -0.05, 0.0, 0.0],
        ]
        
        for sp in start_points:
            if time.time() > deadline - 5.0:
                break
            res_sp = minimize(objective, sp, method='Nelder-Mead', 
                             options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res_sp.fun < best_err:
                best_coeffs = res_sp.x
                best_err = res_sp.fun

    except Exception:
        pass

    return best_coeffs
# EVOLVE-BLOCK-END
