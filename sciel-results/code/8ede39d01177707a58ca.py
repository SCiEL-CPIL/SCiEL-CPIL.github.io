import numpy as np
from scipy.special import lpmv, gamma
from scipy.optimize import minimize
import time

def _build_quadrature():
    """Build 70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    dphi = 2.0 * np.pi / n_phi
    return x_gl, w_gl, phi, dphi

def _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=10):
    """
    Build the matrix Y[i,j] where i indexes quadrature points (n_gl * n_phi)
    and j indexes m = -k..k.
    """
    n_gl = len(x_gl)
    n_phi = len(phi)
    n_m = 2 * k + 1
    Y = np.zeros((n_gl * n_phi, n_m), dtype=np.float64)
    
    for m_idx in range(n_m):
        m = m_idx - k
        abs_m = abs(m)
        # Normalization
        N_m = np.sqrt((2 * k + 1) / (4.0 * np.pi) * 
                       float(gamma(k - abs_m + 1)) / float(gamma(k + abs_m + 1)))
        # Associated Legendre with Condon-Shortley phase
        P = lpmv(abs_m, k, x_gl)  # shape (n_gl,)
        
        if m == 0:
            vals = N_m * P  # shape (n_gl,)
            Y[:, m_idx] = np.repeat(vals, n_phi)
        elif m > 0:
            cos_m = np.cos(m * phi)  # shape (n_phi,)
            vals = np.sqrt(2.0) * N_m * P[:, np.newaxis] * cos_m[np.newaxis, :]
            Y[:, m_idx] = vals.ravel()
        else:
            sin_m = np.sin(abs_m * phi)  # shape (n_phi,)
            vals = np.sqrt(2.0) * N_m * P[:, np.newaxis] * sin_m[np.newaxis, :]
            Y[:, m_idx] = vals.ravel()
    
    return Y

def _compute_ratio(coeffs, Y, weights):
    """Compute ||f||_4 / ||f||_2 using quadrature."""
    f = Y @ coeffs
    f2 = f * f
    f4 = f2 * f2
    norm2_sq = np.dot(weights, f2)
    norm4_sq = np.dot(weights, f4)
    if norm2_sq <= 1e-15:
        return 1e10
    norm2 = np.sqrt(norm2_sq)
    # ||f||_4 = (sum w |f|^4)^(1/4)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    return norm4 / norm2

def _power_iteration_step(coeffs, Y, weights, n_iter=50):
    """
    Perform power iteration on the nonlinear operator T(f) = |f|^2 f.
    This drives the system towards a local minimum of the L4/L2 ratio.
    """
    f = Y @ coeffs
    norm2 = np.sqrt(np.dot(weights, f*f))
    if norm2 < 1e-12:
        return coeffs
    
    f = f / norm2  # Normalize to L2 = 1
    
    for _ in range(n_iter):
        # Nonlinear step: g = |f|^2 f
        g = f * f * f
        
        # Project g back onto the subspace of degree-10 harmonics.
        # We need the coefficients c such that Y @ c approximates g.
        # Since Y columns are orthonormal under the quadrature (approximately, 
        # given the exactness of the quadrature for degree 20 polynomials which 
        # includes products of degree 10 harmonics? 
        # Y_i Y_j is degree 20. 70 GL points are exact for degree 139. 
        # So the Gram matrix Y^T W Y is exactly Identity.
        # Therefore, the projection coefficients are simply Y^T W g.
        
        # g is a vector of size N_points.
        # coeffs_new = Y.T @ (weights * g)
        coeffs_new = Y.T @ (weights * g)
        
        # The eigenvalue equation is Y^T W |Y c|^2 Y c = lambda c.
        # The ratio is related to lambda.
        # We update coeffs to coeffs_new.
        # To maintain stability, we can normalize coeffs_new?
        # The ratio is scale invariant.
        # Let's just use coeffs_new as the next iterate.
        coeffs = coeffs_new
        
    return coeffs

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    k = 10
    x_gl, w_gl, phi, dphi = _build_quadrature()
    Y = _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=k)
    n_gl = len(x_gl)
    n_phi = len(phi)
    weights = np.repeat(w_gl, n_phi) * dphi
    
    # Pre-compute Y.T @ weights (diagonal scaling) for speed?
    # Y.T @ (weights * g) = (Y.T @ diag(weights)) @ g
    # Let W_Y = Y.T @ np.diag(weights). This is a (21 x N) matrix.
    # Or just keep it as Y.T @ (weights * g).
    
    best_coeffs = None
    best_val = np.inf
    
    rng = np.random.default_rng(42)
    
    # Strategy: Multi-start Power Iteration
    # The power iteration converges to a local minimum. 
    # We need diverse starts to find the global minimum.
    
    n_starts = 50
    for i in range(n_starts):
        if time.time() > deadline - 60.0:
            break
            
        # Generate a random start
        # We can bias towards symmetric starts to help convergence
        if i % 2 == 0:
            # Symmetric start: only even m, no sine terms
            c = np.zeros(21)
            # m = 0, 2, 4, 6, 8, 10
            # Indices: 10, 12, 14, 16, 18, 20
            c[10] = rng.normal()
            c[12] = rng.normal()
            c[14] = rng.normal()
            c[16] = rng.normal()
            c[18] = rng.normal()
            c[20] = rng.normal()
        else:
            # Random start
            c = rng.normal(size=21)
            
        # Normalize start to avoid numerical issues
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-12:
            c = np.ones(21)
        else:
            c = c / c_norm
            
        # Run power iteration
        c_iter = _power_iteration_step(c, Y, weights, n_iter=100)
        
        # Compute the ratio
        val = _compute_ratio(c_iter, Y, weights)
        
        if val < best_val:
            best_val = val
            best_coeffs = c_iter.copy()
            
        # If we found a very good value, we can try to polish it with L-BFGS-B
        # But power iteration is usually very good.
        
    # If power iteration didn't find anything good, or if we have time, 
    # polish the best candidate with L-BFGS-B on the full 21D space.
    if best_coeffs is not None and time.time() < deadline - 30.0:
        def objective_full(c):
            return _compute_ratio(c, Y, weights)
        
        # Use the best power iteration result as the initial guess for L-BFGS-B
        # This helps escape shallow local minima that power iteration might get stuck in
        # due to the discrete projection.
        try:
            res = minimize(objective_full, best_coeffs, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except Exception:
            pass

    # If best_coeffs is still None, fallback
    if best_coeffs is None:
        best_coeffs = np.zeros(21)
        best_coeffs[10] = 1.0

    # Final validation and scaling
    best_coeffs = np.asarray(best_coeffs, dtype=np.float64)
    if len(best_coeffs) != 21:
        return np.ones(21)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(21)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        # Scale up
        if max_abs > 0:
            best_coeffs = best_coeffs / max_abs * 1.0
        else:
            best_coeffs[10] = 1.0
    elif max_abs > 1e5:
        # Scale down
        best_coeffs = best_coeffs / max_abs * 1.0

    return best_coeffs
