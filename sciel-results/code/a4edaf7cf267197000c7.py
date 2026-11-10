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

def _compute_grad(coeffs, Y, weights):
    """
    Compute the gradient of R = ||f||_4 / ||f||_2 with respect to coeffs.
    Let N4 = (sum w |f|^4)^{1/4}, N2 = (sum w |f|^2)^{1/2}.
    R = N4 / N2.
    dR/dc = (1/N2) * dN4/dc - (N4/N2^2) * dN2/dc.
    
    N4^4 = sum w f^4.
    4 N4^3 dN4 = sum w 4 f^3 df.
    dN4 = (1/N4^3) sum w f^3 df.
    df = Y dc.
    dN4/dc = (1/N4^3) Y^T (weights * f^3).
    
    N2^2 = sum w f^2.
    2 N2 dN2 = sum w 2 f df.
    dN2 = (1/N2) sum w f df.
    dN2/dc = (1/N2) Y^T (weights * f).
    
    Gradient = (1/N2) * (1/N4^3) Y^T (w f^3) - (N4/N2^2) * (1/N2) Y^T (w f)
             = (1/(N2 N4^3)) Y^T (w f^3) - (N4/(N2^3)) Y^T (w f)
    
    Note: We can simplify by noting that at a minimum, the gradient is zero.
    """
    f = Y @ coeffs
    f2 = f * f
    f3 = f2 * f
    f4 = f2 * f2
    
    norm2_sq = np.dot(weights, f2)
    norm4_sq = np.dot(weights, f4)
    
    if norm2_sq <= 1e-15:
        return np.zeros_like(coeffs)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    
    term1 = Y.T @ (weights * f3)
    term2 = Y.T @ (weights * f)
    
    grad = (term1 / (norm2 * norm4**3)) - (norm4 * term2 / (norm2**3))
    
    return grad

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    k = 10
    x_gl, w_gl, phi, dphi = _build_quadrature()
    Y = _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=k)
    n_gl = len(x_gl)
    n_phi = len(phi)
    weights = np.repeat(w_gl, n_phi) * dphi
    
    best_coeffs = None
    best_val = np.inf
    
    # Define structured initial guesses
    # Indices: m = -10..10. Index i corresponds to m = i - 10.
    # m=0 is index 10.
    # m=10 is index 20.
    # m=-10 is index 0.
    # m=5 is index 15.
    # m=-5 is index 5.
    
    starts = []
    
    # 1. Pure Zonal (m=0)
    c_zonal = np.zeros(21)
    c_zonal[10] = 1.0
    starts.append(c_zonal)
    
    # 2. Pure Sectorial (m=10, cos)
    c_sector = np.zeros(21)
    c_sector[20] = 1.0
    starts.append(c_sector)
    
    # 3. Pure Sectorial (m=-10, sin)
    c_sector_sin = np.zeros(21)
    c_sector_sin[0] = 1.0
    starts.append(c_sector_sin)
    
    # 4. Zonal + Sectorial mix
    c_mix1 = np.zeros(21)
    c_mix1[10] = 1.0
    c_mix1[20] = 1.0
    starts.append(c_mix1)
    
    # 5. Zonal + m=5 mix
    c_mix2 = np.zeros(21)
    c_mix2[10] = 1.0
    c_mix2[15] = 1.0
    starts.append(c_mix2)
    
    # 6. m=10 + m=8 mix
    c_mix3 = np.zeros(21)
    c_mix3[20] = 1.0
    c_mix3[18] = 1.0
    starts.append(c_mix3)
    
    # 7. All even m (cos terms only, m=0,2,4,6,8,10)
    c_even = np.zeros(21)
    for m in [0, 2, 4, 6, 8, 10]:
        c_even[m + 10] = 1.0
    starts.append(c_even)
    
    # 8. All m=0, ±2, ±4, ±6, ±8, ±10 (full even subspace)
    c_even_full = np.zeros(21)
    for m in [0, 2, -2, 4, -4, 6, -6, 8, -8, 10, -10]:
        c_even_full[m + 10] = 1.0
    starts.append(c_even_full)
    
    # 9. Random start with fixed seed
    rng = np.random.default_rng(42)
    c_rand = rng.normal(size=21)
    c_rand /= np.linalg.norm(c_rand)
    starts.append(c_rand)
    
    # 10. Another random start
    c_rand2 = rng.normal(size=21)
    c_rand2 /= np.linalg.norm(c_rand2)
    starts.append(c_rand2)
    
    # Optimize each start using L-BFGS-B
    for i, c_start in enumerate(starts):
        if time.time() > deadline - 20.0:
            break
            
        # Normalize start
        c_start = c_start / np.linalg.norm(c_start)
        
        def objective(c):
            return _compute_ratio(c, Y, weights)
        
        def gradient(c):
            return _compute_grad(c, Y, weights)
        
        try:
            res = minimize(objective, c_start, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except Exception as e:
            pass

    # If we have time, do a few more random starts with L-BFGS-B
    if best_coeffs is not None and time.time() < deadline - 30.0:
        n_extra = 5
        for i in range(n_extra):
            if time.time() > deadline - 10.0:
                break
            c_rand = rng.normal(size=21)
            c_rand /= np.linalg.norm(c_rand)
            
            def objective(c):
                return _compute_ratio(c, Y, weights)
            
            def gradient(c):
                return _compute_grad(c, Y, weights)
            
            try:
                res = minimize(objective, c_rand, jac=gradient, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-10})
                if res.fun < best_val:
                    best_val = res.fun
                    best_coeffs = res.x
            except Exception:
                pass

    # Final validation and scaling
    if best_coeffs is None:
        best_coeffs = np.zeros(21)
        best_coeffs[10] = 1.0

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
