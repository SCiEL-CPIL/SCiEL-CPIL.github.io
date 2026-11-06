"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    """Return six coefficients for the compact growth-index formula.
    
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    where q = 1-Om(a), u = 1+w0+wa*(1-a)
    
    Uses numerical optimization to fit coefficients to the growth index
    across a grid of CPL dark-energy cosmologies.
    """
    deadline = time.time() + 45  # Leave 15s safety margin
    
    # Representative cosmology grid for fitting
    # Om0 in [0.2, 0.4], w0 in [-1.5, -0.5], wa in [-0.5, 0.5]
    Om0_vals = np.array([0.22, 0.27, 0.32, 0.37])
    w0_vals = np.array([-1.2, -1.0, -0.8, -0.6])
    wa_vals = np.array([-0.3, 0.0, 0.3])
    
    # Scale factors for evaluation
    a_vals = np.array([0.3, 0.5, 0.7, 0.85, 1.0])
    
    def Om_a(Om0, w0, wa, a):
        """Matter density parameter at scale factor a."""
        de_factor = (1 - a) ** (-3.0 * (1.0 + w0 + wa))
        if de_factor == 0:
            de_factor = 1e-30
        Om_a_val = Om0 * a**(-3.0) / (Om0 * a**(-3.0) + (1 - Om0) * de_factor)
        return Om_a_val
    
    def gamma_analytic(Om0, w0, wa, a):
        """Approximate analytic growth index for reference."""
        Om = Om_a(Om0, w0, wa, a)
        # Use the Linder approximation: gamma ≈ 0.55 - 0.05 * w_eff
        # where w_eff accounts for the full DE history
        w_eff = w0 + wa * (1 - a)
        # More refined: gamma ≈ (5/9) * Om**0.35 * (1 - 0.5*w_eff)
        gamma = (5.0/9.0) * Om**0.35 * (1.0 - 0.5 * w_eff)
        # Clamp to reasonable range
        gamma = np.clip(gamma, 0.1, 1.2)
        return gamma
    
    def compute_gamma(c, Om0, w0, wa, a):
        """Compute gamma from coefficients."""
        Om = Om_a(Om0, w0, wa, a)
        q = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - a)
        gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
        return gamma
    
    def objective(c):
        """Sum of squared errors in gamma across all test points."""
        errs = []
        for Om0 in Om0_vals:
            for w0 in w0_vals:
                for wa in wa_vals:
                    for a in a_vals:
                        if time.time() > deadline:
                            return 1e10
                        g_model = compute_gamma(c, Om0, w0, wa, a)
                        g_ref = gamma_analytic(Om0, w0, wa, a)
                        # Penalize out-of-range gamma
                        if g_model < 0.1 or g_model > 1.2:
                            errs.append((g_model - 0.65)**2 + 10.0)
                        else:
                            errs.append((g_model - g_ref)**2)
        return np.sum(errs)
    
    # Initial guess based on known growth index behavior
    c0_init = [0.55, -0.05, 0.0, 0.0, 0.0, 0.0]
    
    # Try multiple starting points
    best_c = c0_init
    best_obj = objective(c0_init)
    
    start_points = [
        [0.55, -0.05, 0.0, 0.0, 0.0, 0.0],
        [0.50, -0.10, 0.02, 0.05, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.45, -0.15, 0.05, 0.10, -0.02, 0.0],
    ]
    
    for sp in start_points:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_obj:
                best_obj = res.fun
                best_c = res.x
        except Exception:
            pass
    
    # Ensure coefficients are in [-10, 10]
    best_c = np.clip(best_c, -10.0, 10.0)
    
    return list(best_c)
# EVOLVE-BLOCK-END
