import numpy as np
from scipy.special import lpmv, gamma
from scipy.optimize import differential_evolution, minimize
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

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    k = 10
    x_gl, w_gl, phi, dphi = _build_quadrature()
    Y = _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=k)
    n_gl = len(x_gl)
    n_phi = len(phi)
    weights = np.repeat(w_gl, n_phi) * dphi
    
    def objective(coeffs):
        return _compute_ratio(coeffs, Y, weights)

    best_coeffs = None
    best_val = np.inf
    
    # Strategy: Use a multi-start local optimization approach with random restarts
    # and symmetry-aware initialization to escape local minima
    
    rng = np.random.default_rng(123)
    
    # Generate diverse initial points using different strategies
    def generate_initial_points(n_points):
        points = []
        for i in range(n_points):
            if i % 3 == 0:
                # Pure random
                x0 = rng.normal(size=21)
            elif i % 3 == 1:
                # Sparse: only a few non-zero coefficients
                n_nonzero = rng.integers(1, 6)
                x0 = np.zeros(21)
                indices = rng.choice(21, size=n_nonzero, replace=False)
                x0[indices] = rng.normal(size=n_nonzero)
            else:
                # Symmetry-biased: emphasize certain m values
                x0 = np.zeros(21)
                # Emphasize m=0, ±2, ±4, ±6, ±8, ±10
                for m in [0, 2, -2, 4, -4, 6, -6, 8, -8, 10, -10]:
                    idx = m + 10
                    if rng.random() < 0.5:
                        x0[idx] = rng.normal()
            
            # Normalize
            norm = np.linalg.norm(x0)
            if norm > 1e-10:
                x0 = x0 / norm
            else:
                x0 = rng.normal(size=21)
                x0 = x0 / np.linalg.norm(x0)
            points.append(x0)
        return points
    
    # Phase 1: Multi-start with L-BFGS-B
    n_restarts = 150
    initial_points = generate_initial_points(n_restarts)
    
    for i, x0 in enumerate(initial_points):
        if time.time() > deadline - 60.0:
            break
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            continue
    
    # Phase 2: Refine best solution with tighter tolerance
    if best_coeffs is not None and time.time() < deadline - 30.0:
        try:
            res = minimize(objective, best_coeffs, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                best_coeffs = res.x
                best_val = res.fun
        except Exception:
            pass
    
    # Phase 3: If we have time, do a final DE run from the best neighborhood
    if best_coeffs is not None and time.time() < deadline - 10.0:
        try:
            # Local DE around best solution
            bounds = [(max(-5.0, best_coeffs[i] - 1.0), min(5.0, best_coeffs[i] + 1.0)) 
                      for i in range(21)]
            res = differential_evolution(
                objective, 
                bounds, 
                seed=99,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                polish=True
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
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
