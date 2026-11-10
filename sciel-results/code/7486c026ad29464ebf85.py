# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, gammaln

_DEADLINE = time.time() + 620

def _build_quadrature():
    """70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    cos_theta = x_gl
    phi = np.linspace(0, 2 * np.pi, 130, endpoint=False)
    dphi = 2.0 * np.pi / 130
    weights_1d = w_gl * dphi
    return cos_theta, phi, weights_1d

def _compute_harmonics_matrix(cos_theta, phi):
    """
    Compute the matrix of real spherical harmonics Y_m at all quadrature points.
    Returns H of shape (N_points, 21).
    m ranges from -10 to 10.
    """
    l = 10
    N_theta = len(cos_theta)
    N_phi = len(phi)
    # H shape: (N_theta, N_phi, 21)
    H = np.zeros((N_theta, N_phi, 21), dtype=np.float64)
    
    # Precompute factorials for N_m
    # N_m = sqrt( (2l+1)/(4*pi) * (l-m)! / (l+m)! )
    # Using log for stability
    log_norm_base = 0.5 * (np.log(2*l + 1) - np.log(4.0 * np.pi))
    
    for m_idx, m in enumerate(range(-10, 11)):
        abs_m = abs(m)
        
        # Normalization factor N_m
        log_N = log_norm_base + 0.5 * (gammaln(l - abs_m + 1) - gammaln(l + abs_m + 1))
        N_m = np.exp(log_N)
        
        # Associated Legendre polynomial P_l^m(cos_theta)
        # lpmv includes Condon-Shortley phase (-1)^m
        P = lpmv(abs_m, l, cos_theta) # shape (N_theta,)
        
        if m == 0:
            Y = N_m * P
            H[:, :, m_idx] = Y[:, np.newaxis] * np.ones(N_phi)[np.newaxis, :]
        elif m > 0:
            # Y_m = sqrt(2) N_m P cos(m phi)
            Y = np.sqrt(2.0) * N_m * P
            cos_m_phi = np.cos(m * phi) # shape (N_phi,)
            H[:, :, m_idx] = Y[:, np.newaxis] * cos_m_phi[np.newaxis, :]
        else:
            # Y_m = sqrt(2) N_|m| P sin(|m| phi)
            Y = np.sqrt(2.0) * N_m * P
            sin_abs_m_phi = np.sin(abs_m * phi) # shape (N_phi,)
            H[:, :, m_idx] = Y[:, np.newaxis] * sin_abs_m_phi[np.newaxis, :]
            
    # Flatten to (N_points, 21)
    H_flat = H.reshape(-1, 21)
    return H_flat

def _compute_ratio(coeffs, H_flat, weights_flat):
    """
    Compute ||f||_4 / ||f||_2.
    coeffs: shape (21,)
    H_flat: shape (N_points, 21)
    weights_flat: shape (N_points,)
    """
    f = H_flat @ coeffs
    f2 = f ** 2
    f4 = f2 ** 2
    
    norm2_sq = np.sum(weights_flat * f2)
    norm4_4 = np.sum(weights_flat * f4)
    
    if norm2_sq <= 1e-30:
        return np.inf, np.inf, np.inf
        
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    return norm4 / norm2, norm2_sq, norm4_4

def _objective_and_grad(coeffs, H_flat, weights_flat):
    """
    Minimize ||f||_4^4 subject to ||c||_2 = 1.
    This is equivalent to minimizing ||f||_4 / ||f||_2 when ||f||_2 is fixed to 1.
    """
    f = H_flat @ coeffs
    f2 = f ** 2
    f3 = f2 * f
    f4 = f2 ** 2
    
    # Objective: sum(w * f^4)
    obj = np.sum(weights_flat * f4)
    
    # Gradient of obj w.r.t coeffs
    # d(obj)/d(c_m) = sum_j w_j * 4 * f_j^3 * H_{j,m}
    # grad = 4 * H_flat.T @ (weights_flat * f3)
    grad = 4.0 * (H_flat.T @ (weights_flat * f3))
    
    return obj, grad

def _polish(coeffs, H_flat, weights_flat, max_iter=5000, tol=1e-14):
    """
    Final high-precision polish step using projected gradient descent with adaptive learning rate.
    """
    c = coeffs.copy()
    c /= np.linalg.norm(c)
    
    prev_obj = np.inf
    best_c = c.copy()
    best_obj = np.inf
    lr = 0.01
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        obj, grad = _objective_and_grad(c, H_flat, weights_flat)
        
        if obj < best_obj:
            best_obj = obj
            best_c = c.copy()
        
        # Adaptive learning rate: reduce if objective increases
        c_new = c - lr * grad
        norm_c = np.linalg.norm(c_new)
        if norm_c < 1e-12:
            c_new = c
        else:
            c_new /= norm_c
        
        obj_new, _ = _objective_and_grad(c_new, H_flat, weights_flat)
        
        if obj_new > obj * (1 + 1e-10):
            lr *= 0.5
            if lr < 1e-12:
                break
        else:
            c = c_new
            if abs(prev_obj - obj) < tol:
                break
            prev_obj = obj
    
    return best_c

def _optimize_single_start(c0, H_flat, weights_flat, max_iter=3000):
    """
    Optimize from a single starting point using projected gradient descent with momentum.
    """
    c = c0.copy()
    c /= np.linalg.norm(c)
    momentum = np.zeros_like(c)
    lr = 0.01
    beta = 0.9
    
    prev_obj = np.inf
    best_c = c.copy()
    best_obj = np.inf
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
            
        obj, grad = _objective_and_grad(c, H_flat, weights_flat)
        
        if obj < best_obj:
            best_obj = obj
            best_c = c.copy()
        
        # Momentum update
        momentum = beta * momentum - lr * grad
        c_new = c + momentum
        norm_c = np.linalg.norm(c_new)
        if norm_c < 1e-12:
            c_new = c
            momentum = np.zeros_like(c)
        else:
            c_new /= norm_c
        
        obj_new, _ = _objective_and_grad(c_new, H_flat, weights_flat)
        
        if obj_new > obj * (1 + 1e-10):
            lr *= 0.8
            momentum = beta * momentum
            if lr < 1e-12:
                break
        else:
            c = c_new
            if abs(prev_obj - obj) < 1e-14:
                break
            prev_obj = obj
    
    return best_c

def rudin_problem10() -> np.ndarray:
    """
    Minimize ||f||_4 / ||f||_2 for degree-10 real spherical harmonics.
    Returns 21 coefficients c_m for m = -10..10.
    """
    global _DEADLINE
    
    # 1. Setup
    cos_theta, phi, weights_1d = _build_quadrature()
    N_theta = len(cos_theta)
    N_phi = len(phi)
    
    # Weights for flat array
    weights_flat = np.repeat(weights_1d, N_phi)
    
    # 2. Precompute Harmonics
    H_flat = _compute_harmonics_matrix(cos_theta, phi)
    
    # 3. Optimization
    # We minimize J(c) = sum w_j (H c)_j^4 subject to ||c||_2 = 1.
    # This minimizes the 4th moment. Since ||f||_2 = 1, R = ||f||_4.
    
    best_obj = np.inf
    best_coeffs = None
    
    # Multiple random starts with different strategies
    rng = np.random.default_rng(42)
    
    # Phase 1: Broad search with many random starts
    num_starts_phase1 = 80
    for i in range(num_starts_phase1):
        if time.time() > _DEADLINE - 30:
            break
            
        # Random start on the sphere
        c = rng.standard_normal(21)
        c /= np.linalg.norm(c)
        
        # Optimize
        c_opt = _optimize_single_start(c, H_flat, weights_flat, max_iter=2000)
        
        # Evaluate
        obj, _ = _objective_and_grad(c_opt, H_flat, weights_flat)
        if obj < best_obj:
            best_obj = obj
            best_coeffs = c_opt.copy()
    
    # Phase 2: Refine best solutions
    if best_coeffs is not None and time.time() < _DEADLINE - 60:
        # Polish the best solution
        c_polished = _polish(best_coeffs, H_flat, weights_flat, max_iter=10000, tol=1e-15)
        obj_polished, _ = _objective_and_grad(c_polished, H_flat, weights_flat)
        if obj_polished < best_obj:
            best_obj = obj_polished
            best_coeffs = c_polished
    
    # Phase 3: Try structured starts (symmetric patterns)
    if time.time() < _DEADLINE - 30:
        structured_starts = []
        
        # All zeros except one coefficient
        for m_idx in range(21):
            c = np.zeros(21)
            c[m_idx] = 1.0
            structured_starts.append(c)
        
        # Pairs of symmetric coefficients
        for m_idx in range(10, 21):
            for n_idx in range(m_idx + 1, 21):
                c = np.zeros(21)
                c[m_idx] = 1.0 / np.sqrt(2)
                c[n_idx] = 1.0 / np.sqrt(2)
                structured_starts.append(c)
        
        # Try some structured starts
        num_structured_to_try = min(50, len(structured_starts))
        for i in range(num_structured_to_try):
            if time.time() > _DEADLINE - 15:
                break
                
            c = structured_starts[i]
            c_opt = _optimize_single_start(c, H_flat, weights_flat, max_iter=1500)
            
            obj, _ = _objective_and_grad(c_opt, H_flat, weights_flat)
            if obj < best_obj:
                best_obj = obj
                best_coeffs = c_opt.copy()
    
    # Final polish if time permits
    if best_coeffs is not None and time.time() < _DEADLINE - 10:
        c_polished = _polish(best_coeffs, H_flat, weights_flat, max_iter=20000, tol=1e-16)
        obj_polished, _ = _objective_and_grad(c_polished, H_flat, weights_flat)
        if obj_polished < best_obj:
            best_obj = obj_polished
            best_coeffs = c_polished
    
    # If no good result, fallback
    if best_coeffs is None:
        best_coeffs = np.ones(21) / np.sqrt(21.0)
        
    # Ensure validity
    result = best_coeffs.astype(np.float64)
    if result.shape != (21,):
        result = np.ones(21, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        result = np.ones(21, dtype=np.float64)
        
    max_abs = np.max(np.abs(result))
    if max_abs < 1e-3:
        result = result / max_abs * 0.01 if max_abs > 0 else np.ones(21, dtype=np.float64)
    elif max_abs > 1e5:
        result = result / max_abs * 1e4
        
    return result

# EVOLVE-BLOCK-END
