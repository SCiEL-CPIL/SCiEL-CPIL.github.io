import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

def construct():
    """
    Optimize six coefficients for the growth-index formula.
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    """
    start_time = time.time()
    deadline = 85.0  # Safety margin

    # Define test cosmologies (Om0, w0, wa)
    # Representative set covering typical validation ranges
    test_models = [
        (0.30, 0.0, 0.0),    # LCDM
        (0.30, -1.0, 0.0),   # Cosmological Constant
        (0.30, -0.5, 0.5),   # CPL 1
        (0.30, -0.8, 0.3),   # CPL 2
        (0.30, -0.6, -0.2),  # CPL 3
        (0.25, -0.7, 0.4),   # Low Om, CPL
        (0.35, -1.2, 0.8),   # High Om, w0 < -1
    ]
    
    # Grid for a values
    # Use log spacing for better resolution at early times
    a_grid = np.geomspace(0.01, 1.0, 100)
    
    # Pre-compute reference values for each model
    # Reference: Integrate ODE for D from a=0.01 (matter dominated) to a=1
    # Then normalize D(1)=1 and compute f=D'/D and D_norm
    references = []
    for Om0, w0, wa in test_models:
        def ode_system(a, y):
            D, Dp = y
            # E^2 = Om0 * a^-3 + (1-Om0) * exp(3*wa*(1-a))
            E2 = Om0 * a**-3 + (1-Om0) * np.exp(3*wa*(1-a))
            Om_a = Om0 * a**-3 / E2
            # H'/H = 0.5 * d ln E2 / d ln a
            # d ln E2 / d ln a = a * E2' / E2
            # E2' = -3*Om0*a^-4 + 3*wa*(1-Om0)*exp(3*wa*(1-a))
            E2_prime = -3*Om0*a**-4 + 3*wa*(1-Om0)*np.exp(3*wa*(1-a))
            HpH = 0.5 * a * E2_prime / E2
            
            Dpp = -(3 + HpH) * Dp + 1.5 * Om_a * D
            return [Dp, Dpp]

        # Integrate from a=0.01 to a=1.0
        # Initial conditions at a=0.01: D ~ a, D' ~ 1
        a_start = 0.01
        y0 = [a_start, 1.0]
        
        sol = solve_ivp(ode_system, [a_start, 1.0], y0, method='RK45', rtol=1e-8, atol=1e-10, t_eval=a_grid)
        
        if not sol.success:
            # Fallback if integration fails (unlikely for smooth models)
            continue
            
        D_ref = sol.y[0]
        Dp_ref = sol.y[1]
        
        # Normalize D(1) = 1
        D_ref_norm = D_ref / D_ref[-1]
        Dp_ref_norm = Dp_ref / D_ref[-1]
        
        f_ref = Dp_ref_norm / D_ref_norm
        
        # Store references
        references.append({
            'Om0': Om0,
            'w0': w0,
            'wa': wa,
            'a_grid': a_grid,
            'f_ref': f_ref,
            'D_ref_norm': D_ref_norm
        })

    if not references:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    def compute_error(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        
        for ref in references:
            Om0, w0, wa = ref['Om0'], ref['w0'], ref['wa']
            a_grid = ref['a_grid']
            f_ref = ref['f_ref']
            D_ref_norm = ref['D_ref_norm']
            
            # Compute Om(a) on grid
            E2 = Om0 * a_grid**-3 + (1-Om0) * np.exp(3*wa*(1-a_grid))
            Om_a = Om0 * a_grid**-3 / E2
            q = 1 - Om_a
            u = 1 + w0 + wa * (1 - a_grid)
            
            # Compute gamma
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_grid + c5*u*q
            
            # Check bounds
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e6  # Penalty
            
            # Compute f_approx
            f_approx = Om_a ** gamma
            
            # Compute D_approx_norm
            # D(a)/D(1) = exp( integral from 1 to a of f(e^x) dx )
            # Let h = ln a. dx = dh.
            # Integral from 1 to a is from h=0 to h=ln(a).
            # Since a < 1, ln(a) < 0.
            # We compute cumulative integral of f(a) * (da/a) = f(a) dh
            # We have a_grid. Let's compute dh.
            # h_grid = np.log(a_grid)
            # integrand = f_approx
            # We want integral from 0 to h_i.
            # Since we go from 1 (h=0) down to a_min (h<0), the integral is negative.
            # Let's compute cumulative sum from the end (a=1) backwards.
            
            h_grid = np.log(a_grid)
            # dh is not uniform. Use trapezoidal rule or just simple sum if grid is fine.
            # For 100 points, simple sum of f * dh might be enough, but trapezoid is better.
            # Actually, we can just use np.cumsum on the integrand * dh.
            # But dh varies.
            # Let's use scipy.integrate.cumulative_trapezoid? 
            # To avoid import overhead in loop, we can approximate.
            # Or better, since we have f_approx on a_grid, we can compute the integral directly.
            
            # Let's use a simple cumulative trapezoid implementation for speed
            # integral[i] = integral[i-1] + 0.5 * (f[i] + f[i-1]) * (h[i] - h[i-1])
            # We want integral from 0 (a=1) to h[i].
            # a_grid is increasing, so h_grid is increasing (negative to 0).
            # We want D(a) for a < 1.
            # D(a)/D(1) = exp( int_1^a f da/a ) = exp( - int_a^1 f da/a )
            # Let J(a) = int_a^1 f da/a. This is positive.
            # We compute J on the grid from a=1 downwards.
            
            # Reverse the grid for easier cumulative sum from 1 to a_min
            # a_grid is [0.01, ..., 1.0]
            # f_approx is [f(0.01), ..., f(1.0)]
            
            # Compute integrand = f_approx / a_grid ? No, da/a = dh.
            # So integrand in h-space is just f_approx.
            # We want int_{h_i}^{0} f(e^h) dh.
            
            # Let's compute cumulative integral from the last point (a=1, h=0) backwards.
            # J[ -1 ] = 0
            # J[ -2 ] = 0.5 * (f[-1] + f[-2]) * (h[-1] - h[-2])
            # ...
            
            dh = np.diff(h_grid)
            # f_avg = 0.5 * (f_approx[:-1] + f_approx[1:])
            # dJ = f_avg * dh
            # J_rev = np.concatenate(([0], np.cumsum(dJ)[::-1])) ?
            # Let's do it carefully.
            
            # h_grid: h0 < h1 < ... < hN=0
            # We want J_i = int_{h_i}^{0} f dh
            # J_N = 0
            # J_{N-1} = int_{h_{N-1}}^{h_N} f dh = 0.5 * (f_{N-1} + f_N) * (h_N - h_{N-1})
            # J_{N-2} = J_{N-1} + 0.5 * (f_{N-2} + f_{N-1}) * (h_{N-1} - h_{N-2})
            
            J = np.zeros_like(a_grid)
            # Iterate backwards
            for i in range(len(a_grid)-2, -1, -1):
                J[i] = J[i+1] + 0.5 * (f_approx[i] + f_approx[i+1]) * (h_grid[i+1] - h_grid[i])
                
            D_approx_norm = np.exp(-J)  # Wait.
            # Contract: D(a)/D(1) = exp(- integral_ln(a)^0 f(exp(x)) dx)
            # My J is integral from a to 1.
            # integral_ln(a)^0 is from ln(a) to 0.
            # So yes, D = exp(-J).
            
            # Compute relative errors
            # f error
            err_f = np.abs(f_approx - f_ref) / np.abs(f_ref)
            # D error
            err_D = np.abs(D_approx_norm - D_ref_norm) / np.abs(D_ref_norm)
            
            max_err = max(max_err, np.max(err_f), np.max(err_D))
            
        return max_err

    # Initial guess
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Optimize
    best_x = x0
    best_score = compute_error(x0)
    
    # Use Nelder-Mead
    # Limit iterations to ensure time safety
    max_iter = 200
    
    try:
        result = minimize(compute_error, x0, method='Nelder-Mead', 
                          options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if result.fun < best_score:
            best_x = result.x
            best_score = result.fun
            
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > deadline:
        return best_x.tolist()
    
    # Refine with a second pass if time permits?
    # For now, return the best found.
    
    # Ensure coefficients are within [-10, 10]
    best_x = np.clip(best_x, -10, 10)
    
    return best_x.tolist()
