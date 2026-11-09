"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def construct():
    """Return six coefficients for the growth-index approximation."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Default fallback: constant gamma approximation
    default_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    try:
        # Define helper functions
        def om_matter(a, om0, w0, wa):
            """Matter density parameter at scale factor a."""
            de = om0 * (1.0 - (a**(-3.0 * (1.0 + w0 + wa)))) / (1.0 + w0 + wa) if abs(1.0 + w0 + wa) > 1e-10 else om0 * (1.0 - a**(-3.0)) / 3.0
            de = np.where(np.abs(1.0 + w0 + wa) > 1e-10, 
                          om0 * (1.0 - (a**(-3.0 * (1.0 + w0 + wa)))) / (1.0 + w0 + wa),
                          om0 * (1.0 - a**(-3.0)) / 3.0)
            de = np.clip(de, 0, 1 - om0)
            return 1.0 - de
        
        def h2(a, om0, w0, wa):
            """H^2(a) in units of H0^2."""
            om_m = om_matter(a, om0, w0, wa)
            om_de = 1.0 - om_m
            w = w0 + wa * (1.0 - a)
            return om_m * a**(-3.0) + om_de * a**(-3.0*(1.0+w))
        
        def ode_rhs(x, y, om0, w0, wa):
            """ODE for growth factor: y[0]=D, y[1]=D'"""
            a = np.exp(x)
            h2_val = h2(a, om0, w0, wa)
            h = np.sqrt(h2_val)
            dh_da = 0.5 * h2_val**(-0.5) * np.gradient(h2(a, om0, w0, wa), a) if hasattr(h2_val, 'shape') else 0
            # Use analytical derivative
            om_m = om_matter(a, om0, w0, wa)
            om_de = 1.0 - om_m
            w = w0 + wa * (1.0 - a)
            # d(h2)/da = -3*om_m*a^(-4) + (-3*(1+w))*om_de*a^(-3*(1+w)-1)
            dh2_da = -3.0*om_m*a**(-4.0) + (-3.0*(1.0+w))*om_de*a**(-3.0*(1.0+w)-1.0)
            h_prime = 0.5 * h2_val**(-0.5) * dh2_da
            # D'' + [3/a + h'/h]D' - (3/2)om_m*H0^2/H^2 * D = 0
            # In terms of x=ln(a): d^2D/dx^2 + [3 + (d/da ln H)*a]dD/dx - (3/2)om_m(a) = 0
            dlnH_dx = h_prime / h if h > 1e-10 else 0
            coeff1 = 3.0 + dlnH_dx
            coeff2 = -1.5 * om_m
            return [y[1], -coeff1 * y[1] - coeff2 * y[0]]
        
        def compute_reference(om0, w0, wa, a_vals):
            """Compute reference f and D using ODE."""
            x_vals = np.log(a_vals)
            # Integrate ODE from x=0 (a=1) backwards
            def rhs_back(x, y):
                return ode_rhs(x, y, om0, w0, wa)
            
            # Solve from a=1 to each a
            sol = solve_ivp(rhs_back, [0, min(x_vals)-0.1], [1.0, 0.0], 
                          t_eval=x_vals, method='RK45', rtol=1e-8, atol=1e-10,
                          dense_output=True)
            
            D_ref = sol.y[0]
            Dp_ref = sol.y[1]
            
            # f_ref = D' / D (in terms of ln a)
            f_ref = np.where(np.abs(D_ref) > 1e-10, Dp_ref / D_ref, 0.55)
            
            # Normalized D
            D_norm = D_ref / D_ref[0] if len(D_ref) > 0 and D_ref[0] != 0 else D_ref
            
            return f_ref, D_norm
        
        def compute_approx(coeffs, om0, w0, wa, a_vals):
            """Compute approximate f and D from coefficients."""
            c0, c1, c2, c3, c4, c5 = coeffs
            om_m = om_matter(a_vals, om0, w0, wa)
            q = 1.0 - om_m
            u = 1.0 + w0 + wa * (1.0 - a_vals)
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
            
            # Clamp gamma to [0.1, 1.2]
            gamma = np.clip(gamma, 0.1, 1.2)
            
            f_approx = om_m**gamma
            
            # D(a)/D(1) = exp(-int_{ln a}^0 f(exp(x)) dx)
            x_vals = np.log(a_vals)
            # Integrate f from x to 0
            D_log = np.zeros_like(x_vals)
            for i, x in enumerate(x_vals):
                if x < 0:
                    xs = np.linspace(x, 0, 50)
                    a_int = np.exp(xs)
                    om_int = om_matter(a_int, om0, w0, wa)
                    q_int = 1.0 - om_int
                    u_int = 1.0 + w0 + wa * (1.0 - a_int)
                    gamma_int = c0 + c1*q_int + c2*q_int**2 + c3*u_int + c4*wa*a_int + c5*u_int*q_int
                    gamma_int = np.clip(gamma_int, 0.1, 1.2)
                    f_int = om_int**gamma_int
                    D_log[i] = -np.trapz(f_int, xs)
                else:
                    D_log[i] = 0.0
            
            D_approx = np.exp(D_log)
            
            return f_approx, D_approx
        
        def objective(coeffs):
            """Worst-case relative error across validation models."""
            if time.time() > deadline:
                return 1e10
            
            # Validation cosmologies
            models = [
                (0.3, -1.0, 0.0),   # LCDM
                (0.3, -0.9, 0.1),   # mild wa
                (0.3, -1.1, 0.2),   # more wa
                (0.25, -1.0, 0.0),  # different Om
                (0.35, -1.0, -0.1), # negative wa
            ]
            
            a_vals = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            
            max_err = 0.0
            for om0, w0, wa in models:
                f_ref, D_ref = compute_reference(om0, w0, wa, a_vals)
                f_app, D_app = compute_approx(coeffs, om0, w0, wa, a_vals)
                
                # Relative errors (avoid division by zero)
                mask_f = np.abs(f_ref) > 1e-8
                mask_d = np.abs(D_ref) > 1e-8
                
                if np.any(mask_f):
                    err_f = np.max(np.abs((f_app[mask_f] - f_ref[mask_f]) / f_ref[mask_f]))
                    max_err = max(max_err, err_f)
                
                if np.any(mask_d):
                    err_d = np.max(np.abs((D_app[mask_d] - D_ref[mask_d]) / D_ref[mask_d]))
                    max_err = max(max_err, err_d)
            
            return 100.0 * max_err
        
        # Optimize
        x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Try a few initial points
        best_coeffs = x0
        best_score = 1e10
        
        initial_points = [
            [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.55, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.55, 0.1, 0.0, 0.05, 0.0, 0.0],
            [0.55, 0.05, -0.05, 0.03, -0.02, 0.01],
        ]
        
        for x_init in initial_points:
            if time.time() > deadline - 5.0:
                break
            
            try:
                result = minimize(objective, x_init, method='Nelder-Mead',
                                  options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4,
                                           'adaptive': True})
                
                if result.fun < best_score:
                    best_score = result.fun
                    best_coeffs = result.x
            except Exception:
                continue
        
        # Final check
        if best_score < 1e10:
            # Ensure coefficients are in [-10, 10]
            best_coeffs = np.clip(best_coeffs, -10, 10)
            return [float(c) for c in best_coeffs]
        
    except Exception:
        pass
    
    return default_coeffs
# EVOLVE-BLOCK-END
