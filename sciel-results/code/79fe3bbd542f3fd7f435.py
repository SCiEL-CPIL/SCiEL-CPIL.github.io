"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_legendre


def construct():
    """
    Construct an 8-node positive quadrature rule for the dimensionless
    luminosity distance integral in flat CPL dark-energy models.
    
    The integral is:
      dL/(c/H0) = (1+z) * ∫_0^z z' / E(a(z')) dz'
    where a = 1/(1+z'), E(a) = H(a)/H0.
    
    We change variables to u ∈ [0,1] where z' = z*u, so:
      dL/(c/H0) = (1+z) * z * ∫_0^1 u / E(a(z*u)) du
    
    The quadrature rule must approximate ∫_0^1 f(u) du ≈ Σ w_i f(u_i)
    for f(u) = u / E(a(z*u)) across many z values and many CPL models.
    """
    t_start = time.time()
    DEADLINE = 80.0  # seconds, keep 10s safety margin before 90s kill
    
    # --- Define CPL E(a) models for local optimization ---
    # H(a)/H0 = (Omega_m/a^3 + Omega_r/a^4 + Omega_L * a^(-3(1+w0+w1*(1-a))))^(1/2)
    # In terms of z: a = 1/(1+z)
    # E(z) = (Omega_m*(1+z)^3 + Omega_r*(1+z)^4 + Omega_L*(1+z)^(3*(1+w0+w1*z)))^(1/2)
    
    def make_E(Omega_m, Omega_r, Omega_L, w0, w1):
        """Return a function E(z) = H(z)/H0 for given CPL parameters."""
        def E(z):
            return np.sqrt(
                Omega_m * (1.0 + z)**3 +
                Omega_r * (1.0 + z)**4 +
                Omega_L * (1.0 + z)**(3.0 * (1.0 + w0 + w1 * z))
            )
        return E
    
    # Representative CPL models for local design
    models = [
        (0.3, 1e-5, 0.7, -1.0, 0.0),      # Lambda CDM
        (0.3, 1e-5, 0.7, -0.5, -0.5),     # CPL with w0=-0.5, w1=-0.5
        (0.3, 1e-5, 0.7, -1.5, 1.0),      # Phantom-ish
        (0.3, 1e-5, 0.7, -0.9, -0.3),     # Mild CPL
        (0.3, 1e-5, 0.65, -1.0, 0.3),     # Different Omega_L
        (0.25, 1e-5, 0.75, -1.0, 0.0),    # Different Omega_m
        (0.3, 1e-5, 0.7, -1.0, -0.5),     # w1 negative
        (0.3, 1e-5, 0.7, -0.7, 0.7),      # Another CPL
    ]
    
    # Test redshifts for local error evaluation
    z_tests = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0])
    
    def objective(params):
        """
        params: 16 values = [u_0,...,u_7, w_0,...,w_7]
        Minimize worst-case relative error over models and z tests.
        """
        if time.time() - t_start > DEADLINE:
            return 1e10
        
        u = np.array(params[:8])
        w = np.array(params[8:16])
        
        # Enforce ordering: u must be strictly increasing in [0,1]
        # Use a soft penalty approach
        diffs = np.diff(u)
        if np.any(diffs <= 1e-10):
            return 1e10
        if u[0] < -1e-10 or u[-1] > 1.0 + 1e-10:
            return 1e10
        if np.any(w <= 0):
            return 1e10
        
        # Clamp u to [0,1]
        u = np.clip(u, 1e-8, 1.0 - 1e-8)
        
        max_rel_err = 0.0
        for Om, Or, Ol, w0, w1 in models:
            E_func = make_E(Om, Or, Ol, w0, w1)
            for z in z_tests:
                # Integrand f(u) = u / E(z*u)
                # Exact integral via high-resolution quadrature
                u_fine = np.linspace(0, 1, 10000)
                f_fine = u_fine / E_func(z * u_fine)
                exact = np.trapz(f_fine, u_fine)
                
                # Quadrature approximation
                f_quad = u / E_func(z * u)
                approx = np.dot(w, f_quad)
                
                if abs(exact) > 1e-15:
                    rel_err = abs(approx - exact) / abs(exact)
                    if rel_err > max_rel_err:
                        max_rel_err = rel_err
        
        return max_rel_err
    
    # --- Initialize with Gauss-Legendre nodes ---
    gl_nodes, gl_weights = roots_legendre(8)
    # Map from [-1,1] to [0,1]
    u0 = 0.5 * (gl_nodes + 1.0)
    w0 = 0.5 * gl_weights
    
    x0 = np.concatenate([u0, w0])
    
    # --- Local optimization using L-BFGS-B ---
    best_params = x0.copy()
    best_obj = objective(x0)
    
    bounds = [(1e-8, 1.0 - 1e-8)] * 8 + [(1e-8, None)] * 8
    
    # Try a few local optimization methods
    for method in ['L-BFGS-B', 'Nelder-Mead']:
        if time.time() - t_start > DEADLINE - 5:
            break
        
        try:
            if method == 'L-BFGS-B':
                res = minimize(objective, x0, method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12})
            else:
                res = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # --- Differential evolution for global search if time permits ---
    if time.time() - t_start < DEADLINE - 15:
        try:
            de_bounds = [(1e-6, 1.0 - 1e-6)] * 8 + [(1e-6, 1.0)] * 8
            res_de = differential_evolution(
                objective, de_bounds, seed=42, maxiter=50,
                popsize=15, tol=1e-8, polish=False,
                init='sobol'
            )
            if res_de.fun < best_obj:
                best_obj = res_de.fun
                best_params = res_de.x.copy()
        except Exception:
            pass
    
    # --- Final local polish ---
    if time.time() - t_start < DEADLINE - 3:
        try:
            res_final = minimize(objective, best_params, method='L-BFGS-B',
                                 bounds=bounds, options={'maxiter': 200, 'ftol': 1e-14})
            if res_final.fun < best_obj:
                best_obj = res_final.fun
                best_params = res_final.x.copy()
        except Exception:
            pass
    
    u_final = np.clip(best_params[:8], 1e-8, 1.0 - 1e-8)
    w_final = np.clip(best_params[8:16], 1e-12, None)
    
    # Enforce strictly increasing nodes
    for i in range(1, 8):
        if u_final[i] <= u_final[i-1] + 1e-12:
            u_final[i] = u_final[i-1] + 1e-12
    # Re-clamp to [0,1]
    u_final = np.clip(u_final, 1e-8, 1.0 - 1e-8)
    
    # Normalize weights to sum to 1
    w_sum = np.sum(w_final)
    if w_sum > 0:
        w_final = w_final / w_sum
    else:
        w_final = np.full(8, 1.0/8)
    
    # Final validation
    assert len(u_final) == 8
    assert len(w_final) == 8
    assert np.all(np.diff(u_final) > 0)
    assert np.all(u_final >= 0) and np.all(u_final <= 1)
    assert np.all(w_final > 0)
    assert abs(np.sum(w_final) - 1.0) < 1e-12
    
    return {"nodes": u_final, "weights": w_final}
# EVOLVE-BLOCK-END
