# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, Bounds

# Precompute quadrature nodes and weights (70-point Gauss-Legendre in cos theta, 130 phi)
def _setup_quadrature():
    n_theta = 70
    n_phi = 130
    # Gauss-Legendre nodes and weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # phi grid
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # Total weight for each (theta, phi) point: w_i * dphi (since dOmega = dphi * d(cos theta))
    # Actually the surface element is dOmega = sin(theta) dtheta dphi = dphi * d(mu) where mu = cos(theta)
    # So the weight is w_i * dphi
    weights = np.outer(w, np.full(n_phi, dphi))
    # Reshape to 2D arrays
    mu = np.tile(x[:, None], (1, n_phi))  # shape (70, 130)
    phi_grid = np.tile(phi[None, :], (n_theta, 1))  # shape (70, 130)
    
    return mu, phi_grid, weights.flatten(), x, w, phi, dphi


def _real_harmonics(k, mu, phi):
    """
    Compute real orthonormal spherical harmonics of degree k at given mu=cos(theta), phi.
    Returns array of shape (2k+1, N) where N is number of points.
    Order: m = -k, -k+1, ..., 0, ..., k-1, k
    """
    N = mu.size
    mu_flat = mu.reshape(-1)
    phi_flat = phi.reshape(-1)
    
    Y = np.zeros((2 * k + 1, N), dtype=np.float64)
    
    for m in range(-k, k + 1):
        idx = m + k
        abs_m = abs(m)
        # Normalization N_m = sqrt((2k+1)/(4*pi) * (k-m)!/(k+m)!)
        # For real harmonics, the normalization is slightly different
        # Y_0 = N_0 P_k^0
        # Y_m = sqrt(2) N_m P_k^m cos(m phi) for m > 0
        # Y_m = sqrt(2) N_|m| P_k^|m| sin(|m| phi) for m < 0
        # where N_m = sqrt((2k+1)/(4*pi) * (k-m)!/(k+m)!)
        
        # Compute associated Legendre polynomial P_k^m(cos(theta)) with Condon-Shortley phase
        # scipy.special.lpmv(m, k, mu) includes the Condon-Shortley phase (-1)^m
        P = lpmv(abs_m, k, mu_flat)
        
        norm = np.sqrt((2 * k + 1) / (4 * np.pi) * factorial(k - abs_m) / factorial(k + abs_m))
        
        if m == 0:
            Y[idx, :] = norm * P
        elif m > 0:
            Y[idx, :] = np.sqrt(2) * norm * P * np.cos(m * phi_flat)
        else:
            Y[idx, :] = np.sqrt(2) * norm * P * np.sin(abs_m * phi_flat)
    
    return Y


def _compute_ratio(coeffs, Y):
    """
    Compute ||f||_4 / ||f||_2 where f = sum c_m Y_m.
    Y has shape (2k+1, N).
    """
    f = Y.T @ coeffs  # shape (N,)
    f2 = f ** 2
    f4 = f ** 4
    
    # Norms with surface measure (weights already include dOmega)
    # But we need to be careful: the quadrature approximates integral over sphere
    # ||f||_2^2 = integral f^2 dOmega ≈ sum_i w_i f_i^2
    # ||f||_4^4 = integral f^4 dOmega ≈ sum_i w_i f_i^4
    
    # For the ratio, we need:
    # R = (integral f^4 dOmega)^(1/4) / (integral f^2 dOmega)^(1/2)
    
    return f2, f4


def _objective_and_grad(coeffs, Y, weights):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient.
    """
    f = Y.T @ coeffs
    f2 = f ** 2
    f4 = f ** 4
    
    # Weighted integrals
    int_f2 = np.sum(weights * f2)
    int_f4 = np.sum(weights * f4)
    
    if int_f2 <= 0:
        return np.inf, np.zeros_like(coeffs)
    
    # R = (int_f4)^(1/4) / (int_f2)^(1/2)
    # Let's minimize R^4 / R_2^2 = int_f4 / int_f2^2 (monotonic transformation)
    # Actually, let's just minimize R directly or a smooth surrogate
    
    R = (int_f4 ** 0.25) / (int_f2 ** 0.5)
    
    # Gradient of R with respect to coeffs
    # R = A^(1/4) / B^(1/2) where A = int_f4, B = int_f2
    # dR/dc = (1/4) A^(-3/4) B^(-1/2) dA/dc - (1/2) A^(1/4) B^(-3/2) dB/dc
    # dA/dc = sum_i w_i 4 f_i^3 Y_i
    # dB/dc = sum_i w_i 2 f_i Y_i
    
    dA_dc = Y @ (4 * weights * f ** 3)  # shape (2k+1,)
    dB_dc = Y @ (2 * weights * f)  # shape (2k+1,)
    
    grad_R = (0.25 * A ** (-0.75) * B ** (-0.5)) * dA_dc - (0.5 * A ** 0.25 * B ** (-1.5)) * dB_dc
    
    # Wait, I need to be more careful with the notation. Let me redo this.
    # A = int_f4, B = int_f2
    # R = A^(1/4) * B^(-1/2)
    # dR/dcoeff_j = (1/4) A^(-3/4) B^(-1/2) * dA/dcoeff_j + A^(1/4) * (-1/2) B^(-3/2) * dB/dcoeff_j
    
    dA_dc = Y @ (4 * weights * f ** 3)
    dB_dc = Y @ (2 * weights * f)
    
    grad_R = (0.25 * (int_f4 ** (-0.75)) * (int_f2 ** (-0.5))) * dA_dc \
           + (int_f4 ** 0.25) * (-0.5 * (int_f2 ** (-1.5))) * dB_dc
    
    return R, grad_R


def _objective_only(coeffs, Y, weights):
    f = Y.T @ coeffs
    f2 = f ** 2
    f4 = f ** 4
    int_f2 = np.sum(weights * f2)
    int_f4 = np.sum(weights * f4)
    if int_f2 <= 0:
        return np.inf
    return (int_f4 ** 0.25) / (int_f2 ** 0.5)


def rudin_problem10() -> np.ndarray:
    k = 10
    n_coeffs = 2 * k + 1
    
    start_time = time.time()
    deadline = start_time + 600  # 10 minutes
    
    # Setup quadrature
    mu, phi_grid, weights, x, w, phi, dphi = _setup_quadrature()
    
    # Precompute spherical harmonics
    Y = _real_harmonics(k, mu, phi_grid)
    
    # Initial guess: all ones
    best_coeffs = np.ones(n_coeffs, dtype=np.float64)
    best_R = _objective_only(best_coeffs, Y, weights)
    
    # Try multiple starting points and optimization methods
    rng = np.random.RandomState(42)
    
    # Generate diverse initial points
    initial_points = [np.ones(n_coeffs, dtype=np.float64)]
    
    # Random initial points
    for _ in range(20):
        init = rng.randn(n_coeffs)
        init /= np.linalg.norm(init)
        initial_points.append(init)
    
    # Also try some structured initial points
    for m0 in range(-k, k + 1):
        init = np.zeros(n_coeffs, dtype=np.float64)
        init[m0 + k] = 1.0
        initial_points.append(init)
    
    # Pairs of modes
    for m1 in range(-k, k + 1, 2):
        for m2 in range(m1 + 2, k + 1, 2):
            init = np.zeros(n_coeffs, dtype=np.float64)
            init[m1 + k] = 1.0
            init[m2 + k] = 1.0
            initial_points.append(init)
    
    # Optimize from each initial point
    for i, init in enumerate(initial_points):
        if time.time() > deadline - 30:
            break
        
        try:
            # Use L-BFGS-B with bounds
            res = minimize(
                lambda c: _objective_only(c, Y, weights),
                init,
                method='L-BFGS-B',
                bounds=[(-100, 100)] * n_coeffs,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.success or res.fun < best_R:
                R_new = res.fun
                if R_new < best_R:
                    best_R = R_new
                    best_coeffs = res.x.copy()
                    
        except Exception:
            continue
    
    # Try trust-region method on a few of the best points
    for i, init in enumerate(initial_points[:10]):
        if time.time() > deadline - 30:
            break
        
        try:
            res = minimize(
                lambda c: _objective_only(c, Y, weights),
                init,
                method='trust-constr',
                bounds=Bounds(-100, 100),
                options={'maxiter': 200, 'gtol': 1e-10}
            )
            
            if res.success or res.fun < best_R:
                R_new = res.fun
                if R_new < best_R:
                    best_R = R_new
                    best_coeffs = res.x.copy()
                    
        except Exception:
            continue
    
    # Final polish with tight tolerance
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda c: _objective_only(c, Y, weights),
                best_coeffs,
                method='L-BFGS-B',
                bounds=[(-100, 100)] * n_coeffs,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_R:
                best_R = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Validate result
    if np.any(~np.isfinite(best_coeffs)):
        best_coeffs = np.ones(n_coeffs, dtype=np.float64)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs / max_abs * 1e-1 if max_abs > 0 else np.ones(n_coeffs)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs * 1e5
    
    return best_coeffs


# EVOLVE-BLOCK-END
