"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _hermite(n, x):
    """Compute physicists' Hermite polynomial H_n(x) via recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    H0 = np.ones_like(x)
    H1 = 2 * x
    for k in range(1, n):
        H2 = 2 * x * H1 - 2 * k * H0
        H0, H1 = H1, H2
    return H1

def _compute_C4(c0, c4, c8):
    """Compute the upper bound C4 given coefficients c0, c4, c8.
    c12 is determined by g(0) = 0 constraint."""
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    n_quad = 200
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(n_quad)
    
    H0 = _hermite(0, x_nodes)
    H4 = _hermite(4, x_nodes)
    H8 = _hermite(8, x_nodes)
    H12 = _hermite(12, x_nodes)
    
    P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g_sq = P ** 2
    
    moments = np.zeros(16)
    for m in range(16):
        moments[m] = np.sum(w_nodes * (x_nodes ** m) * g_sq)
    
    P_prime = c4 * 8 * _hermite(3, x_nodes) + c8 * 16 * _hermite(7, x_nodes) + c12 * 24 * _hermite(11, x_nodes)
    P_double_prime = c4 * 48 * _hermite(2, x_nodes) + c8 * 224 * _hermite(6, x_nodes) + c12 * 528 * _hermite(10, x_nodes)
    
    g_prime_sq = (P_prime - x_nodes * P) ** 2
    integral_g_prime_sq = np.sum(w_nodes * g_prime_sq)
    
    g_double_prime_sq = (P_double_prime - 2 * x_nodes * P_prime + (x_nodes**2 - 1) * P) ** 2
    integral_g_double_prime_sq = np.sum(w_nodes * g_double_prime_sq)
    
    integral_g_sq = np.sum(w_nodes * g_sq)
    
    moment2_x = moments[2] / moments[0] if moments[0] != 0 else 0
    moment4_x = moments[4] / moments[0] if moments[0] != 0 else 0
    
    C4_candidate4 = (integral_g_double_prime_sq / moments[0]) * (moments[2] / moments[0])
    
    return C4_candidate4

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # safety margin before 180s kill
    
    seed_coeffs = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_C4 = _compute_C4(*seed_coeffs)
    
    best_C4 = seed_C4
    best_coeffs = seed_coeffs.copy()
    
    # Use spherical coordinates to remove scale ambiguity
    # Let c0 = r * sin(phi) * cos(theta), c4 = r * sin(phi) * sin(theta), c8 = r * cos(phi)
    # This reduces the search to 2D (theta, phi) with r fixed or optimized separately
    
    def _objective_spherical(params):
        theta, phi = params
        r = 1.0  # We can optimize r later if needed
        c0 = r * np.sin(phi) * np.cos(theta)
        c4 = r * np.sin(phi) * np.sin(theta)
        c8 = r * np.cos(phi)
        return _compute_C4(c0, c4, c8)
    
    # First, do a coarse search in spherical coordinates
    if time.time() < deadline - 60:
        bounds = [(-np.pi, np.pi), (0, np.pi)]
        try:
            result_de = differential_evolution(_objective_spherical, bounds,
                                               maxiter=200, popsize=15,
                                               seed=42, tol=1e-10,
                                               init='sobol')
            if result_de.success and result_de.fun < best_C4:
                theta_best, phi_best = result_de.x
                r = 1.0
                best_coeffs = np.array([r * np.sin(phi_best) * np.cos(theta_best),
                                        r * np.sin(phi_best) * np.sin(theta_best),
                                        r * np.cos(phi_best)])
                best_C4 = result_de.fun
        except Exception:
            pass
    
    # Refine with Nelder-Mead in original coordinates
    if time.time() < deadline - 20:
        try:
            result = minimize(_compute_C4, best_coeffs, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            if result.success and result.fun < best_C4:
                best_C4 = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            result = minimize(_compute_C4, best_coeffs, method='Nelder-Mead',
                            options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16})
            if result.success and result.fun < best_C4:
                best_C4 = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
