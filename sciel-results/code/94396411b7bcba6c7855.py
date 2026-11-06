"""Improved six-coefficient growth-index approximation for CPL dark-energy cosmologies."""
import numpy as np
from scipy.integrate import solve_ivp
import time

# EVOLVE-BLOCK-START
def construct():
    """Return six finite coefficients in [-10,10] for the growth-index ansatz.
    
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    where q = 1-Om(a), u = 1+w0+wa*(1-a), Om(a) is matter density fraction.
    f = Om(a)^gamma, D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
    """
    deadline = time.time() + 80  # Safety margin before 90s limit
    
    def Om_a(a, Om0, w0, wa):
        """Matter density fraction as function of scale factor."""
        if a <= 0:
            return Om0
        # For CPL: w(a) = w0 + wa*(1-a)
        # Omega_DE(a) ∝ a^{-3(1+w0+wa)} * exp(3*wa*(a-1))
        # Omega_m(a) = Om0 * a^{3*wa} * exp(-3*wa*(a-1)) / (Om0 * a^{3*wa} * exp(-3*wa*(a-1)) + (1-Om0))
        # Wait, let's re-derive carefully.
        # rho_m ∝ a^{-3}
        # rho_de ∝ a^{-3(1+w0)} * exp(3*wa*(a-1))  ... actually:
        # d(rho_de)/da = -3(1+w(a))*rho_de/a
        # ln(rho_de) = -3∫(1+w(a'))/a' da' = -3[(1+w0)ln(a) + wa(a-1)]
        # So rho_de ∝ a^{-3(1+w0)} * exp(-3*wa*(a-1))
        # Om_de(a) = (1-Om0) * a^{-3(1+w0)} * exp(-3*wa*(a-1)) / [Om0*a^{-3} + (1-Om0)*a^{-3(1+w0)}*exp(-3*wa*(a-1))]
        # Multiply num and den by a^3:
        # Om_m(a) = Om0 / [Om0 + (1-Om0)*a^{-3*w0} * exp(-3*wa*(a-1))]
        
        de_factor = (1-Om0) * np.exp(-3*w0*np.log(a) - 3*wa*(a-1))
        return Om0 / (Om0 + de_factor)
    
    def gr_reference(a_arr, Om0, w0, wa):
        """Compute reference f and D using the GR ODE."""
        a_arr = np.atleast_1d(a_arr)
        
        def ode_rhs(ln_a, state):
            a_local = np.exp(ln_a)
            Om_local = Om_a(a_local, Om0, w0, wa)
            
            # E^2(a) = Om_m(a) * a^{-3} + Om_de(a)
            # For flat universe: E^2 = Om0*a^{-3} + (1-Om0)*a^{-3(1+w0)}*exp(-3*wa*(a-1))
            # Let's use the analytical form for E^2
            # E^2 = Om0*a^{-3} + (1-Om0)*exp(-3*w0*ln(a) - 3*wa*(a-1))
            
            E2 = Om0 * a_local**(-3) + (1-Om0) * np.exp(-3*w0*np.log(a_local) - 3*wa*(a_local-1))
            
            # d(ln E^2)/d(ln a) = a * E2' / E2
            # E2' = -3*Om0*a^{-4} + (1-Om0)*exp(-3*w0*ln(a) - 3*wa*(a-1)) * (-3*w0/a - 3*wa)
            E2_prime = -3*Om0*a_local**(-4) + (1-Om0)*np.exp(-3*w0*np.log(a_local) - 3*wa*(a_local-1)) * (-3*w0/a_local - 3*wa)
            dlnE2_dlna = a_local * E2_prime / E2
            
            # Growth ODE in terms of f = dln(D)/dln(a):
            # f' + f^2 + [3 + 1/2*dln(E^2)/dln(a)]*f - 3/2*Om_m = 0
            # where ' denotes d/dln(a)
            
            coeff = 3.0 + 0.5 * dlnE2_dlna
            return [state[1], -state[1]**2 - coeff*state[1] + 1.5*Om_local]
        
        # Integrate the Riccati equation for f backward from a=1
        # At a=1, f(1) = Om0^gamma(1) ≈ 0.55 (use the approximation as initial guess)
        # Actually, we should integrate the D equation for stability
        
        def ode_rhs_D(ln_a, state):
            a_local = np.exp(ln_a)
            Om_local = Om_a(a_local, Om0, w0, wa)
            E2 = Om0 * a_local**(-3) + (1-Om0) * np.exp(-3*w0*np.log(a_local) - 3*wa*(a_local-1))
            E2_prime = -3*Om0*a_local**(-4) + (1-Om0)*np.exp(-3*w0*np.log(a_local) - 3*wa*(a_local-1)) * (-3*w0/a_local - 3*wa)
            dlnE2_dlna = a_local * E2_prime / E2
            coeff = 3.0 + 0.5 * dlnE2_dlna
            return [state[1], -coeff*state[1] + 1.5*Om_local*state[0]]
        
        a_min = 0.05
        ln_a_vals = np.linspace(0.0, np.log(a_min), 100)
        
        try:
            sol = solve_ivp(ode_rhs_D, [0.0, np.log(a_min)], [1.0, 0.55], 
                           t_eval=ln_a_vals, rtol=1e-8, atol=1e-10, method='DOP853')
            
            if not sol.success:
                # Fallback
                Om_ref = Om_a(a_arr, Om0, w0, wa)
                f_ref = Om_ref**0.55
                D_ref = np.ones_like(a_arr)
                return f_ref, D_ref
            
            # Interpolate
            D_interp = np.interp(np.log(a_arr), sol.t, sol.y[0])
            Dp_interp = np.interp(np.log(a_arr), sol.t, sol.y[1])
            f_ref = Dp_interp
            D_ref = D_interp  # D(1)=1 by initial condition
            return f_ref, D_ref
            
        except Exception:
            Om_ref = Om_a(a_arr, Om0, w0, wa)
            f_ref = Om_ref**0.55
            D_ref = np.ones_like(a_arr)
            return f_ref, D_ref
    
    def compute_f_D(a_arr, c, Om0, w0, wa):
        """Compute f and normalized D for a range of a values."""
        Om = Om_a(a_arr, Om0, w0, wa)
        q = 1.0 - Om
        u = 1.0 + w0 + wa*(1.0 - a_arr)
        gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a_arr + c[5]*u*q
        gamma = np.clip(gamma, 0.1, 1.2)
        f = Om**gamma
        
        # Compute D(a)/D(1) = exp(-∫_{ln(a)}^0 f(e^x) dx)
        D_norm = np.ones_like(a_arr)
        for i, ai in enumerate(a_arr):
            if ai == 1.0:
                D_norm[i] = 1.0
            else:
                ln_ai = np.log(ai)
                # ∫_{ln(a)}^0 f(e^x) dx
                # For a < 1: ln(a) < 0, integral is from negative to 0
                # For a > 1: ln(a) > 0, integral is from positive to 0 (negative)
                n_pts = 30
                if ai < 1.0:
                    x_vals = np.linspace(ln_ai, 0.0, n_pts)
                else:
                    x_vals = np.linspace(0.0, ln_ai, n_pts)
                    # ∫_{ln(a)}^0 = -∫_0^{ln(a)}
                    # So D/D(1) = exp(-(-∫_0^{ln(a)} f dx)) = exp(∫_0^{ln(a)} f dx)
                    # Wait, let's be consistent.
                    # D(a)/D(1) = exp(-∫_{ln(a)}^0 f(e^x) dx)
                    # If a>1: ∫_{ln(a)}^0 = -∫_0^{ln(a)}, so D/D(1) = exp(∫_0^{ln(a)} f(e^x) dx)
                    # If a<1: ∫_{ln(a)}^0 is positive (since ln(a)<0 and f>0), so D/D(1) = exp(-positive) < 1
                    # Let's just compute the integral directly with proper sign handling.
                
                # Direct computation
                if ai < 1.0:
                    x_vals = np.linspace(ln_ai, 0.0, n_pts)
                    a_vals = np.exp(x_vals)
                    Om_vals = Om_a(a_vals, Om0, w0, wa)
                    q_vals = 1.0 - Om_vals
                    u_vals = 1.0 + w0 + wa*(1.0 - a_vals)
                    gamma_vals = c[0] + c[1]*q_vals + c[2]*q_vals**2 + c[3]*u_vals + c[4]*wa*a_vals + c[5]*u_vals*q_vals
                    gamma_vals = np.clip(gamma_vals, 0.1, 1.2)
                    f_vals = Om_vals**gamma_vals
                    integral = np.trapz(f_vals, x_vals)
                    D_norm[i] = np.exp(-integral)
                else:
                    x_vals = np.linspace(0.0, ln_ai, n_pts)
                    a_vals = np.exp(x_vals)
                    Om_vals = Om_a(a_vals, Om0, w0, wa)
                    q_vals = 1.0 - Om_vals
                    u_vals = 1.0 + w0 + wa*(1.0 - a_vals)
                    gamma_vals = c[0] + c[1]*q_vals + c[2]*q_vals**2 + c[3]*u_vals + c[4]*wa*a_vals + c[5]*u_vals*q_vals
                    gamma_vals = np.clip(gamma_vals, 0.1, 1.2)
                    f_vals = Om_vals**gamma_vals
                    integral = np.trapz(f_vals, x_vals)
                    # ∫_{ln(a)}^0 f dx = -∫_0^{ln(a)} f dx = -integral
                    # D/D(1) = exp(-(-integral)) = exp(integral)
                    D_norm[i] = np.exp(integral)
        
        return f, D_norm
    
    # Define test cosmologies for fitting
    # Broad range to capture validation set behavior
    test_models = [
        (0.25, -1.0, 0.0),
        (0.3, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        (0.3, -1.0, 0.5),
        (0.3, -1.0, -0.5),
        (0.3, -0.8, 0.3),
        (0.3, -0.8, -0.3),
        (0.25, -1.2, 0.3),
        (0.35, -0.8, 0.2),
        (0.3, -1.0, 1.0),
        (0.2, -1.5, 0.5),
        (0.4, -0.9, -0.3),
        (0.3, -1.0, 0.8),
        (0.3, -1.0, -0.8),
    ]
    
    a_test = np.linspace(0.05, 0.99, 15)
    
    # Pre-compute references
    references = {}
    for Om0, w0, wa in test_models:
        if time.time() > deadline:
            break
        f_ref, D_ref = gr_reference(a_test, Om0, w0, wa)
        references[(Om0, w0, wa)] = (f_ref, D_ref)
    
    if not references:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def objective(c):
        if time.time() > deadline:
            return 1e10 * np.ones(len(a_test) * len(test_models) * 2)
        
        residuals = []
        for Om0, w0, wa in test_models:
            if (Om0, w0, wa) not in references:
                continue
            f_ref, D_ref = references[(Om0, w0, wa)]
            f_approx, D_approx = compute_f_D(a_test, c, Om0, w0, wa)
            
            # Relative errors
            f_rel = (f_approx - f_ref) / np.abs(f_ref)
            D_rel = (D_approx - D_ref) / np.abs(D_ref)
            residuals.extend(f_rel)
            residuals.extend(D_rel)
        
        return np.array(residuals)
    
    # Initial guess: constant gamma
    c0_init = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_c = c0_init.copy()
    best_cost = np.inf
    
    try:
        from scipy.optimize import least_squares
        result = least_squares(objective, c0_init, bounds=([-10]*6, [10]*6), 
                               method='trf', max_nfev=200, xtol=1e-8, ftol=1e-8)
        if np.all(np.isfinite(result.x)):
            best_c = np.clip(result.x, -10, 10)
            best_cost = result.cost
    except Exception:
        pass
    
    # Verify gamma constraint across a wider range
    def check_gamma(c):
        for Om0, w0, wa in test_models:
            for a in np.linspace(0.01, 1.0, 30):
                Om = Om_a(a, Om0, w0, wa)
                q = 1.0 - Om
                u = 1.0 + w0 + wa*(1.0 - a)
                g = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
                if g < 0.1 or g > 1.2:
                    return False
        return True
    
    if not check_gamma(best_c):
        # Try to adjust by reducing coefficients
        for scale in [0.5, 0.25, 0.1]:
            adjusted = best_c * scale
            if check_gamma(adjusted):
                best_c = adjusted
                break
        else:
            best_c = c0_init.copy()
    
    return [float(x) for x in best_c]
# EVOLVE-BLOCK-END
