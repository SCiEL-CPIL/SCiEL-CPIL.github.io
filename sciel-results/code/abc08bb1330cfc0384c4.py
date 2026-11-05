"""Rewritten program for the suite task; optimized for speed and accuracy."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

def _precompute_references(cosmologies, a_grid, deadline):
    """
    Pre-compute reference f_ref and D_ref for given cosmologies and a_grid.
    Returns a list of dictionaries, one per cosmology, containing a_grid, f_ref, D_ref.
    """
    refs = []
    for cosmo in cosmologies:
        Om0, w0, wa = cosmo
        
        # Define ODE for D(x) where x = ln(a)
        # D'' + (3/x + H'/H) D' - 1.5 * Om(x) * D = 0
        def growth_ode(x, y):
            a_x = np.exp(x)
            if a_x < 1e-6:
                a_x = 1e-6
            
            # E^2(a) = Om0 * a^-3 + (1-Om0) * a^(-3(1+w0+wa))
            E2 = Om0 * a_x**(-3) + (1 - Om0) * a_x**(-3 * (1 + w0 + wa))
            
            # d ln E^2 / d ln a
            # Numerator: -3 * Om0 * a^-3 - 3(1+w0+wa) * (1-Om0) * a^(-3(1+w0+wa))
            num = -3 * (Om0 * a_x**(-3) + (1 + w0 + wa) * (1 - Om0) * a_x**(-3 * (1 + w0 + wa)))
            dlnE2_dlna = num / E2
            
            Hprime_over_H = 0.5 * dlnE2_dlna
            
            # Om(x)
            Om_x = Om0 * a_x**(-3) / E2
            
            D, Dprime = y
            # Avoid division by zero or negative x issues at very small a
            # x is ln(a), so for a < 1, x < 0.
            # The term 3/x is singular at x=0 (a=1), but we integrate up to x=0.
            # Actually, the standard form is valid for x != 0.
            # At a=1, x=0, the equation has a singularity in the 3/x term if not handled.
            # However, we integrate from small a to a=1.
            # Let's check the limit. As a->1, x->0.
            # It is safer to integrate from a=1 to small a? No, initial conditions are easier at small a.
            # The singularity at x=0 is removable if D' is finite.
            # To be safe, we integrate from a=1e-4 to a=0.999 and extrapolate or just stop at 0.999?
            # The task asks for a in (0, 1).
            # Let's integrate from x_min = ln(1e-4) to x=0.
            # If x is very close to 0, 3/x is large.
            # Actually, the growth equation in terms of a is:
            # D''_a + (3/a + H'/H) D'_a - 1.5 Om D = 0
            # Let x = ln a. D_a = D_x / a. D_aa = (D_xx - D_x)/a^2.
            # (D_xx - D_x)/a^2 + (3/a + H'/H) D_x/a - 1.5 Om D = 0
            # Multiply by a^2:
            # D_xx - D_x + (3a + a^2 H'/H) D_x/a - 1.5 Om a^2 D = 0 ?? No.
            # Let's stick to x = ln a.
            # d/da = (1/a) d/dx.
            # d^2/da^2 = (1/a) d/dx (1/a d/dx) = (1/a) (-1/a^2 d/dx + 1/a d^2/dx^2) = (1/a^2) (D_xx - D_x).
            # Eq: (1/a^2)(D_xx - D_x) + (3/a + H'/H) (1/a) D_x - 1.5 Om D = 0.
            # Multiply by a^2:
            # D_xx - D_x + (3a + a^2 H'/H) (D_x/a) - 1.5 Om a^2 D = 0
            # D_xx - D_x + (3 + a H'/H) D_x - 1.5 Om a^2 D = 0 ??
            # Wait, H'/H in x-domain is d ln H / dx.
            # Let's use the standard x-form directly:
            # D_xx + (3/x + d ln H / dx) D_x - 1.5 Om D = 0 is INCORRECT for x=ln a.
            
            # Correct derivation:
            # D(a) = D(x).
            # D'_a = (1/a) D_x.
            # D''_a = (1/a^2) (D_xx - D_x).
            # H(a) = H(x). H'_a = (1/a) H_x.
            # H'/H = (1/a) H_x / H = (1/a) (d ln H / dx).
            
            # Original: D''_a + (3/a + H'_a/H) D'_a - 1.5 Om D = 0
            # (1/a^2)(D_xx - D_x) + (3/a + (1/a) dlnH/dx) (1/a D_x) - 1.5 Om D = 0
            # Multiply by a^2:
            # (D_xx - D_x) + (3 + dlnH/dx) D_x - 1.5 Om a^2 D = 0 ??
            # No, the last term is -1.5 Om D. It doesn't have a factor of a^2.
            # So: D_xx - D_x + (3 + dlnH/dx) D_x - 1.5 Om D = 0
            # D_xx + (2 + dlnH/dx) D_x - 1.5 Om D = 0.
            
            # Let's re-verify.
            # Standard result: D'' + (3/x + H'/H) D' ... is often seen for D(z) or similar.
            # Let's trust the derivation: D_xx + (2 + dlnH/dx) D_x - 1.5 Om D = 0.
            
            # dlnH/dx = dlnH/dlna = 0.5 * dlnE2/dlna.
            
            dlnH_dx = 0.5 * dlnE2_dlna
            
            coeff_Dx = 2.0 + dlnH_dx
            coeff_D = -1.5 * Om_x
            
            Ddoubleprime = -coeff_Dx * Dprime - coeff_D * D
            return [Dprime, Ddoubleprime]

        # Initial conditions at a_min = 1e-4
        a_min = 1e-4
        x_min = np.log(a_min)
        D_min = a_min
        Dprime_min = 1.0
        
        # Integrate to x=0 (a=1)
        # We need values on the grid a_grid.
        # a_grid is likely in (0, 1).
        # Let's integrate from x_min to 0 and use dense output or solve for specific points.
        # Since a_grid is fixed, we can solve_ivp with t_eval=a_grid mapped to x.
        
        x_grid = np.log(a_grid)
        # Ensure x_min is smaller than min(x_grid)
        if x_min > np.min(x_grid):
            x_min = np.min(x_grid) - 0.1
            a_min = np.exp(x_min)
            D_min = a_min
            Dprime_min = 1.0
            
        # Sort x_grid for integration
        # a_grid is usually increasing, so x_grid is increasing.
        # We integrate from x_min to 0.
        # We need values at x_grid.
        
        try:
            sol = solve_ivp(growth_ode, [x_min, 0], [D_min, Dprime_min], 
                            method='RK45', rtol=1e-9, atol=1e-12, 
                            t_eval=np.sort(x_grid), dense_output=True)
            if not sol.success:
                continue
                
            D_vals = sol.y[0]
            Dprime_vals = sol.y[1]
            
            # Normalize D such that D(1) = 1.
            # Find index closest to x=0
            # t_eval was sorted x_grid. x=0 is the max if a_grid < 1.
            # Actually, a_grid might not include 1.
            # We need D_ref(a) = D(a) / D(1).
            # We can interpolate D at x=0 from the dense output.
            D_at_1 = sol.sol(0)[0]
            
            D_norm = D_vals / D_at_1
            f_ref = -Dprime_vals / D_vals
            
            # Store results
            refs.append({
                'a': a_grid,
                'f_ref': f_ref,
                'D_ref': D_norm,
                'Om0': Om0,
                'w0': w0,
                'wa': wa
            })
        except Exception:
            continue
            
    return refs

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Training cosmologies covering a reasonable range
    cosmologies = [
        (0.27, -1.0, 0.0),
        (0.30, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        (0.30, -1.0, 0.3),
        (0.30, -1.0, -0.5),
        (0.30, -0.9, 0.1),
        (0.30, -1.1, -0.2),
        (0.25, -1.0, 0.5),
    ]
    
    # Grid for evaluation. Include points close to 0 and 1 to capture behavior.
    # Log-spaced is better for a in (0,1)
    a_grid = np.logspace(-2, -0.01, 50)  # 0.01 to ~0.98
    
    # Pre-compute references
    refs = _precompute_references(cosmologies, a_grid, deadline)
    
    if not refs:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    def objective(coeffs):
        if time.time() > deadline:
            return 1e10
            
        c0, c1, c2, c3, c4, c5 = coeffs
        
        max_err = 0.0
        penalty = 0.0
        
        for ref in refs:
            a = ref['a']
            f_ref = ref['f_ref']
            D_ref = ref['D_ref']
            Om0 = ref['Om0']
            w0 = ref['w0']
            wa = ref['wa']
            
            # Compute Om(a)
            a_inv3 = a ** (-3)
            a_inv3_cpl = a ** (-3 * (1 + w0 + wa))
            denom = Om0 * a_inv3 + (1 - Om0) * a_inv3_cpl
            Om_a = Om0 * a_inv3 / denom
            q = 1 - Om_a
            u = 1 + w0 + wa * (1 - a)
            
            # Compute gamma
            gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
            
            # Check constraints
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                penalty += 1e6 * np.mean(np.abs((gamma < 0.1) + (gamma > 1.2)))
            
            # Compute f_approx
            f_approx = Om_a ** gamma
            
            # Error in f
            err_f = np.abs(f_approx - f_ref) / np.abs(f_ref)
            max_err = max(max_err, np.max(err_f))
            
            # Compute D_approx
            # D_approx(a) = exp(-int_{ln a}^0 f_approx(x) dx)
            # We have a grid of x = ln(a).
            x = np.log(a)
            # Integrate f_approx from x to 0.
            # Since x is negative (a<1), integral from x to 0 is positive if f>0.
            # We can use cumulative trapezoidal integration from the end (x=0 side) backwards.
            # Or simply: I(x_i) = -int_{x_i}^0 f dx = int_0^{x_i} f dx (with sign flip)
            # Let's compute cumulative integral from 0 to x_i.
            # np.cumtrapz(f, x, initial=0) gives int_0^{x_i} f dx.
            # Since x_i < 0, this integral is negative.
            # int_{x_i}^0 f dx = - int_0^{x_i} f dx.
            # So D_approx = exp( - ( - cumtrapz ) ) = exp( cumtrapz ).
            
            # Ensure x is sorted increasing for cumtrapz
            # a_grid is increasing, so x is increasing.
            # x[0] is min (most negative), x[-1] is max (closest to 0).
            # We want integral from x_i to 0.
            # Let J(x) = int_0^x f(t) dt.
            # J(0) = 0.
            # J(x_i) = int_0^{x_i} f(t) dt.
            # Since x_i < 0, this is - int_{x_i}^0 f(t) dt.
            # So int_{x_i}^0 f(t) dt = -J(x_i).
            # D_approx = exp( - (-J(x_i)) ) = exp(J(x_i)).
            
            # Compute J(x_i) using cumtrapz.
            # np.cumtrapz(y, x, initial=0) computes integral from x[0] to x[i].
            # We want integral from 0 to x[i].
            # Let's reverse the array, integrate, and reverse back.
            # Or simpler:
            # Integral from x[0] to x[i] is known.
            # Total integral from x[0] to x[-1] (approx 0) is I_total.
            # J(x_i) = I_total - Integral from x_i to x[-1].
            # This is getting complicated.
            
            # Easier:
            # Create a grid that includes 0.
            # But 0 is not in a_grid (a<1).
            # We can append x=0, f=f(1).
            # f(1) = Om(1)^gamma(1). Om(1)=Om0.
            
            x_ext = np.append(x, 0.0)
            f_ext = np.append(f_approx, Om0 ** gamma[-1]) # Approximation of f at a=1
            
            # Integrate from x_ext[0] to x_ext[i]
            # We want integral from 0 to x_i.
            # Let's compute cumulative integral from the last point (0) backwards.
            # Or just use scipy.integrate.cumulative_trapezoid on the reversed array.
            
            # Reverse x and f
            x_rev = x_ext[::-1]
            f_rev = f_ext[::-1]
            
            # Integrate from 0 to x_i (which is integrating backwards in x_rev)
            # cumtrapz(f_rev, x_rev, initial=0) gives integral from 0 to x_rev[k].
            # x_rev[0] = 0.
            # x_rev[k] = x_ext[-k].
            
            try:
                from scipy.integrate import cumulative_trapezoid
                J_rev = cumulative_trapezoid(f_rev, x_rev, initial=0.0)
                # J_rev[k] is integral from 0 to x_rev[k].
                # We need J(x_i) for x_i in x.
                # x corresponds to x_ext[:-1].
                # x_ext[:-1] reversed is x_rev[1:].
                # So J(x_i) = J_rev[1:][::-1].
                J_vals = J_rev[1:][::-1]
                
                D_approx = np.exp(J_vals)
                
                # Error in D
                err_D = np.abs(D_approx - D_ref) / np.abs(D_ref)
                max_err = max(max_err, np.max(err_D))
            except Exception:
                max_err = 1e10
                
        return max_err + penalty

    # Initial guess
    best_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_err = objective(best_coeffs)
    
    # Optimization
    # Use Nelder-Mead for robustness with constraints (via penalty)
    try:
        result = minimize(objective, best_coeffs, method='Nelder-Mead', 
                         options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
        if result.fun < best_err:
            best_coeffs = result.x
            best_err = result.fun
    except Exception:
        pass
    
    # Clip coefficients
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return list(best_coeffs)
