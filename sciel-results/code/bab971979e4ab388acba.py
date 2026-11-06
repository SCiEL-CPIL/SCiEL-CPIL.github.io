"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """Return six coefficients for the compact growth-index formula.
    
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    where q = 1-Om(a), u = 1+w0+wa*(1-a)
    
    Optimizes on f errors with a better reference and more cosmologies.
    """
    deadline = time.time() + 78
    
    def Om_m(Om0, w0, wa, a):
        if abs(wa) > 1e-10:
            de_term = (1 - a**(-3*(1+w0+wa))) / (3*wa)
        else:
            de_term = 1 - a**3
        return Om0 * a**3 / (Om0 * a**3 + (1 - Om0) * de_term)
    
    def gamma_from_coeffs(c, Om0, w0, wa, a):
        Om = Om_m(Om0, w0, wa, a)
        q = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - a)
        return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
    
    def f_from_gamma(gamma, Om):
        return Om**gamma
    
    def gamma_ref(Om0, w0, wa, a):
        Om = Om_m(Om0, w0, wa, a)
        gamma = 0.55 - 0.12*(1.0 - Om0) - 0.1*(w0 - 1.0) + 0.05*wa + 0.03*(Om - Om0)
        return np.clip(gamma, 0.1, 1.2)
    
    cosmo_grid = [
        (0.30, -1.0, 0.0), (0.30, -0.9, 0.0), (0.30, -1.0, 0.3),
        (0.30, -1.1, -0.3), (0.25, -1.0, 0.0), (0.35, -1.0, 0.0),
        (0.30, -0.8, 0.2), (0.30, -1.2, 0.5), (0.30, -1.0, -0.5),
        (0.28, -0.95, 0.1), (0.32, -1.05, -0.2), (0.30, -1.0, 0.1),
    ]
    
    a_vals = np.array([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0])
    
    ref_data = []
    for Om0, w0, wa in cosmo_grid:
        for a in a_vals:
            Om = Om_m(Om0, w0, wa, a)
            g_ref = gamma_ref(Om0, w0, wa, a)
            f_ref = f_from_gamma(g_ref, Om)
            ref_data.append((Om0, w0, wa, a, Om, g_ref, f_ref))
    
    def objective(c):
        f_errors = []
        for Om0, w0, wa, a, Om, g_ref, f_ref in ref_data:
            g_app = gamma_from_coeffs(c, Om0, w0, wa, a)
            if g_app < 0.1 or g_app > 1.2:
                return 1e6
            f_app = f_from_gamma(g_app, Om)
            if abs(f_ref) > 1e-10:
                f_errors.append(abs(f_app - f_ref) / abs(f_ref))
        
        if not f_errors:
            return 1e6
        return max(f_errors)
    
    c_init = np.array([0.55, -0.12, 0.05, -0.05, 0.0, 0.0])
    best_c = c_init.copy()
    best_obj = objective(best_c)
    
    from scipy.optimize import minimize
    
    try:
        result = minimize(objective, c_init, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9,
                                  'adaptive': True})
        if time.time() < deadline and result.fun < best_obj:
            best_c = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        best_c = np.clip(best_c, -10.0, 10.0)
        return [float(x) for x in best_c]
    
    try:
        result2 = minimize(objective, best_c, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
        if time.time() < deadline and result2.fun < best_obj:
            best_c = result2.x
            best_obj = result2.fun
    except Exception:
        pass
    
    best_c = np.clip(best_c, -10.0, 10.0)
    return [float(x) for x in best_c]
# EVOLVE-BLOCK-END
