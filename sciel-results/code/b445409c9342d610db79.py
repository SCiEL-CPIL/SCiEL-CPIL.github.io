# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv, gamma
from scipy.optimize import basinhopping, minimize
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
    norm4 = np.sqrt(norm4_sq)
    return norm4 / norm2

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    k = 10
    x_gl, w_gl, phi, dphi = _build_quadrature()
    Y = _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=k)
    n_gl = len(x_gl)
    n_phi = len(phi)
    weights = np.repeat(w_gl, n_phi) * dphi
    
    # Precompute Y.T @ Y? No, we need f^4.
    
    def objective(coeffs):
        # We can normalize coeffs here to avoid scaling issues, 
        # but the ratio is scale invariant, so it's fine.
        return _compute_ratio(coeffs, Y, weights)

    # Initial guess: Zonal harmonic (m=0) is often a good starting point for "flatness"
    # Or a random vector.
    # Let's use a few diverse initial points.
    
    best_coeffs = None
    best_val = np.inf
    
    # Strategy: Basin Hopping
    # Local minimizer: L-BFGS-B
    # Step size: 1.0
    
    # We will run basin hopping for a fixed number of iterations or until time runs out.
    # To be safe, we'll run multiple short basin hopping sessions from different starts.
    
    rng = np.random.default_rng(42)
    
    n_starts = 3
    for i in range(n_starts):
        if time.time() > deadline - 60.0:
            break
            
        if i == 0:
            x0 = np.zeros(21)
            x0[10] = 1.0 # m=0
        elif i == 1:
            x0 = np.zeros(21)
            x0[0] = 1.0 # m=-10
        else:
            x0 = rng.normal(size=21)
            x0 /= np.linalg.norm(x0)
        
        # Run basin hopping
        # We limit the local minimization steps to save time
        local_min_kwargs = {'method': 'L-BFGS-B', 'maxiter': 100}
        
        try:
            res = basinhopping(
                objective, 
                x0, 
                niter=20, # Moderate number of global steps
                stepsize=1.5,
                minimizer_kwargs=local_min_kwargs,
                seed=42+i
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
                
        except Exception as e:
            print(f"Basin hopping failed: {e}")
            continue
            
        # Check time
        if time.time() > deadline - 30.0:
            break
            
    # Fallback if basin hopping didn't find anything good or failed
    if best_coeffs is None or not np.isfinite(best_val):
        # Fall back to a simple Nelder-Mead on a good start
        x0 = np.zeros(21)
        x0[10] = 1.0
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000})
        best_coeffs = res.x
        best_val = res.fun

    # Final polish with Nelder-Mead on the best found so far
    if time.time() < deadline - 10.0:
        res = minimize(objective, best_coeffs, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_val:
            best_coeffs = res.x
            best_val = res.fun

    # Validate
    best_coeffs = np.asarray(best_coeffs, dtype=np.float64)
    if len(best_coeffs) != 21:
        return np.ones(21)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(21)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Normalize to valid range if possible
        if max_abs > 0:
            best_coeffs = best_coeffs / max_abs * 1.0
        else:
            return np.ones(21)
            
    return best_coeffs

# EVOLVE-BLOCK-END
