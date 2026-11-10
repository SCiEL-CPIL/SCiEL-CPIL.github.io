# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import lpmv

def _build_basis(k=10, n_theta=70, n_phi=130):
    """Build the real orthonormal spherical harmonic basis values on the grid."""
    # Gauss-Legendre quadrature in x = cos(theta)
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # x in [-1, 1], w are weights
    # theta = arccos(x), but we work in x directly
    # phi equally spaced
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    # Create grid
    X, Phi = np.meshgrid(x, phi, indexing='ij')  # shape (n_theta, n_phi)
    
    # Quadrature weights: dOmega = dphi * dx (since sin(theta)dtheta = -dx)
    # Total integral = sum w_i * (2*pi/n_phi) for each phi point... 
    # Actually: integral over sphere = (1/n_phi) * sum_phi sum_i w_i * f(x_i, phi_j) * (2*pi)
    # Wait: dOmega = dphi dx, so integral = sum_j (dphi) sum_i w_i f(x_i, phi_j)
    # where dphi = 2*pi/n_phi
    dphi = 2 * np.pi / n_phi
    # weights for each grid point
    W = np.outer(w, np.full(n_phi, dphi))  # shape (n_theta, n_phi)
    
    # Build basis functions Y_m for m = -k..k
    # Y_0 = N_0 * P_k^0(cos theta)
    # Y_m = sqrt(2) * N_m * P_k^m(cos theta) * cos(m*phi) for m > 0
    # Y_m = sqrt(2) * N_|m| * P_k^{|m|}(cos theta) * sin(|m|*phi) for m < 0
    # N_m = sqrt(2k+1/(4*pi) * (k-m)!/(k+m)!)
    
    basis = np.zeros((2*k+1, n_theta, n_phi), dtype=np.float64)
    
    for m_idx, m in enumerate(range(-k, k+1)):
        am = abs(m)
        # Condon-Shortley phase is included in lpmv
        P = lpmv(am, k, X)  # P_k^am(x) with Condon-Shortley phase
        N_m = np.sqrt((2*k+1)/(4*np.pi) * np.math.factorial(k-am) / np.math.factorial(k+am))
        
        if m == 0:
            basis[m_idx] = N_m * P
        elif m > 0:
            basis[m_idx] = np.sqrt(2) * N_m * P * np.cos(m * Phi)
        else:
            basis[m_idx] = np.sqrt(2) * N_m * P * np.sin(am * Phi)
    
    return basis, W

def _evaluate_ratio(c, basis, W):
    """Compute ||f||_4 / ||f||_2 for coefficients c."""
    # f = sum c_m * basis[m]
    f = np.tensordot(c, basis, axes=([0],[0]))  # shape (n_theta, n_phi)
    
    f2 = f * f
    f4 = f2 * f2
    
    # L2 norm squared = integral f^2 dOmega = sum W * f2
    l2_sq = np.sum(W * f2)
    if l2_sq < 1e-15:
        return np.inf
    
    # L4 norm = integral f^4 dOmega = sum W * f4
    l4 = np.sum(W * f4)
    
    # ||f||_4 / ||f||_2 = (l4)^(1/4) / (l2_sq)^(1/2)
    ratio = (l4 ** 0.25) / (l2_sq ** 0.5)
    return ratio

def _objective(c, basis, W):
    """Objective: minimize ||f||_4^4 / ||f||_2^8 to avoid root issues, 
       but actually let's minimize the ratio directly or a monotonic function."""
    # To make it differentiable and smooth, minimize ||f||_4^4 / ||f||_2^8
    # which is proportional to ratio^4
    f = np.tensordot(c, basis, axes=([0],[0]))
    f2 = f * f
    f4 = f2 * f2
    
    l2_sq = np.sum(W * f2)
    if l2_sq < 1e-15:
        return 1e10
    l4 = np.sum(W * f4)
    
    # ratio^4 = l4 / l2_sq^2
    return l4 / (l2_sq * l2_sq)

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    """
    start_time = time.time()
    deadline = start_time + 570  # 570s safety margin from 630s limit
    
    k = 10
    basis, W = _build_basis(k=k, n_theta=70, n_phi=130)
    
    # Initial guess: all ones
    c0 = np.ones(2*k+1, dtype=np.float64)
    
    # Normalize initial guess
    f0 = np.tensordot(c0, basis, axes=([0],[0]))
    l2_sq0 = np.sum(W * f0 * f0)
    if l2_sq0 > 1e-15:
        c0 = c0 / np.sqrt(l2_sq0)
    
    # Try multiple initial points and local optimization
    best_c = c0.copy()
    best_val = _objective(c0, basis, W)
    
    # Generate diverse initial points
    rng = np.random.default_rng(42)
    
    n_starts = 0
    max_starts = 30
    
    for start_idx in range(max_starts):
        if time.time() > deadline:
            break
        
        if start_idx == 0:
            c_init = c0.copy()
        else:
            # Random initialization with various structures
            strategy = start_idx % 5
            if strategy == 0:
                c_init = rng.standard_normal(2*k+1)
            elif strategy == 1:
                # Emphasize m=0
                c_init = rng.standard_normal(2*k+1)
                c_init[10] = 3.0
            elif strategy == 2:
                # Emphasize high |m|
                c_init = rng.standard_normal(2*k+1)
                c_init[0] = 3.0
                c_init[20] = 3.0
            elif strategy == 3:
                # Symmetric pattern
                c_init = np.zeros(2*k+1)
                c_init[10] = 1.0
                c_init[5] = rng.uniform(0.5, 1.5)
                c_init[15] = rng.uniform(0.5, 1.5)
            else:
                # Cosine-like pattern
                m_vals = np.arange(-k, k+1)
                c_init = np.cos(m_vals * rng.uniform(0, np.pi)) + 0.1 * rng.standard_normal(2*k+1)
            
            # Normalize
            f_init = np.tensordot(c_init, basis, axes=([0],[0]))
            l2_sq_init = np.sum(W * f_init * f_init)
            if l2_sq_init > 1e-15:
                c_init = c_init / np.sqrt(l2_sq_init)
        
        n_starts += 1
        
        # Local optimization using L-BFGS-B
        try:
            result = minimize(
                _objective,
                c_init,
                args=(basis, W),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.success or result.fun < best_val:
                c_opt = result.x
                val = result.fun
                if val < best_val:
                    best_val = val
                    best_c = c_opt.copy()
        except Exception:
            pass
    
    # Additional refinement with more iterations on the best point
    if time.time() < deadline - 60:
        try:
            result = minimize(
                _objective,
                best_c,
                args=(basis, W),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    # Final check: ensure validity
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        best_c = c0.copy()
    
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c / max_abs * 1e-2
    elif max_abs > 1e5:
        best_c = best_c / max_abs * 1e4
    
    return best_c

# EVOLVE-BLOCK-END
