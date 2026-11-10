# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv
import time


def _build_quadrature():
    """Build 70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    # Gauss-Legendre nodes and weights for 70 points on [-1, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    # x_gl = cos(theta), w_gl are weights
    # phi: 130 equally spaced points
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    # Surface element: dOmega = dphi * d(cos theta), so weight for each (i,j) is w_gl[i] * (2*pi/n_phi)
    dphi = 2 * np.pi / n_phi
    # Create meshgrid
    ct, ph = np.meshgrid(x_gl, phi, indexing='ij')
    # Weights for quadrature
    w = np.outer(w_gl, np.full(n_phi, dphi))
    return ct, ph, w


def _spherical_harmonics_real(ct, ph, k=10):
    """
    Compute real orthonormal spherical harmonics Y_m for m = -k..k.
    Y_0 = N_0 P^0
    Y_m = sqrt(2) N_m P^m cos(m*phi) for m > 0
    Y_m = sqrt(2) N_|m| P^|m| sin(|m|*phi) for m < 0
    where P^m = P_k^m(cos theta) with Condon-Shortley phase
    N_m = sqrt(21/(4*pi) * (k-m)!/(k+m)!)
    
    Returns array of shape (2k+1, n_theta, n_phi)
    """
    n = 2 * k + 1
    shape = ct.shape
    Y = np.zeros((n,) + shape, dtype=np.float64)
    
    for m in range(-k, k + 1):
        abs_m = abs(m)
        # Condon-Shortley phase: P_k^m includes (-1)^m
        # lpmv includes Condon-Shortley phase
        P = lpmv(abs_m, k, np.arccos(np.clip(ct, -1, 1)))
        
        # Normalization N_m
        # N_m = sqrt((2k+1)/(4*pi) * (k-m)!/(k+m)!)
        # For m=0: N_0 = sqrt((2k+1)/(4*pi))
        # For m>0: the real harmonics use sqrt(2)*N_m where N_m uses (k-m)!/(k+m)!
        # Actually the normalization given is N_m = sqrt(21/(4*pi) * (k-m)!/(k+m)!)
        # with 2k+1 = 21 for k=10
        
        log_num = np.sum(np.log(np.arange(1, k - abs_m + 1))) if k - abs_m > 0 else 0.0
        log_den = np.sum(np.log(np.arange(1, k + abs_m + 1))) if k + abs_m > 0 else 0.0
        N_m = np.sqrt((2 * k + 1) / (4 * np.pi) * np.exp(log_num - log_den))
        
        idx = m + k  # index in array: m=-10 -> 0, m=0 -> 10, m=10 -> 20
        
        if m == 0:
            Y[idx] = N_m * P
        elif m > 0:
            Y[idx] = np.sqrt(2) * N_m * P * np.cos(m * ph)
        else:
            Y[idx] = np.sqrt(2) * N_m * P * np.sin(abs_m * ph)
    
    return Y


def _compute_R(c, Y, w):
    """Compute R = ||f||_4 / ||f||_2 given coefficients c and precomputed harmonics."""
    # f = sum c_m Y_m
    f = np.tensordot(c, Y, axes=([0], [0]))
    
    # ||f||_2^2 = sum w * f^2
    norm2_sq = np.sum(w * f * f)
    if norm2_sq <= 0:
        return np.inf
    
    # ||f||_4^4 = sum w * f^4, so ||f||_4 = (sum w * f^4)^(1/4)
    norm4_4 = np.sum(w * f * f * f * f)
    norm4 = norm4_4 ** 0.25
    
    norm2 = np.sqrt(norm2_sq)
    
    return norm4 / norm2


def _objective(c, Y, w):
    """Objective function: R^2 to avoid square roots (monotonic with R)."""
    f = np.tensordot(c, Y, axes=([0], [0]))
    norm2_sq = np.sum(w * f * f)
    if norm2_sq <= 0:
        return 1e10
    norm4_4 = np.sum(w * f * f * f * f)
    # R^2 = norm4_4 / norm2_sq
    return norm4_4 / norm2_sq


def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline
    
    k = 10
    n = 2 * k + 1  # 21 coefficients
    
    # Build quadrature and spherical harmonics
    ct, ph, w = _build_quadrature()
    Y = _spherical_harmonics_real(ct, ph, k)
    
    # Verify orthonormality
    # <Y_i, Y_j> = sum w * Y_i * Y_j should be delta_ij
    # Y has shape (21, 70, 130)
    # Inner product: sum over theta, phi of w * Y_i * Y_j
    Y_flat = Y.reshape(n, -1)
    w_flat = w.flatten()
    gram = Y_flat @ np.diag(w_flat) @ Y_flat.T
    # Should be identity
    
    # Initial guess: all ones
    c0 = np.ones(n, dtype=np.float64)
    
    # Normalize initial guess
    c0 = c0 / np.linalg.norm(c0)
    
    best_c = c0.copy()
    best_val = _objective(c0, Y, w)
    
    # Try multiple initializations
    np.random.seed(42)
    
    # Run optimization from multiple starting points
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        
        if trial == 0:
            c_init = c0
        else:
            c_init = np.random.randn(n)
            c_init = c_init / np.linalg.norm(c_init)
        
        # Use L-BFGS-B with constraint ||c|| = 1 handled by re-normalizing
        # Actually, since R is scale invariant, we can just minimize R^2 directly
        # and the optimizer will find the right scale
        
        def obj(c_vec):
            return _objective(c_vec, Y, w)
        
        try:
            result = minimize(
                obj,
                c_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if result.success and result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
                
        except Exception:
            pass
    
    # Additional refinement with Nelder-Mead on the best solution
    if time.time() < deadline - 30:
        try:
            def obj2(c_vec):
                return _objective(c_vec, Y, w)
            
            result2 = minimize(
                obj2,
                best_c,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            
            if result2.fun < best_val:
                best_val = result2.fun
                best_c = result2.x.copy()
                
        except Exception:
            pass
    
    # Validate result
    if not np.all(np.isfinite(best_c)):
        best_c = c0
    
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c / max_abs * 1.0
    if max_abs > 1e5:
        best_c = best_c / max_abs * 1.0
    
    return best_c


# EVOLVE-BLOCK-END
