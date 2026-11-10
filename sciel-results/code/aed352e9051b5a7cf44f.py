# EVOLVE-BLOCK-START
import numpy as np
import time

def _gauss_legendre(n):
    """Return nodes and weights for n-point Gauss-Legendre on [-1, 1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w

def _spherical_harmonics_real(l, m_vals, cos_theta, phi):
    """
    Compute real orthonormal spherical harmonics Y_l^m for given m values.
    Y_0 = N_0 P_l^0(cos theta)
    Y_m = sqrt(2) N_m P_l^m(cos theta) cos(m*phi) for m > 0
    Y_m = sqrt(2) N_|m| P_l^|m|(cos theta) sin(|m|*phi) for m < 0
    where N_m = sqrt((2l+1)/(4*pi) * (l-m)!/(l+m)!)
    """
    from scipy.special import lpmv, factorial
    
    l_val = l
    n_pts = len(cos_theta)
    n_phi = len(phi)
    
    # Create meshgrid
    ct, ph = np.meshgrid(cos_theta, phi, indexing='ij')
    
    results = {}
    for m in m_vals:
        am = abs(m)
        # Normalization
        norm = np.sqrt((2*l_val + 1) / (4 * np.pi) * 
                       factorial(l_val - am, exact=True) / 
                       factorial(l_val + am, exact=True))
        
        # Associated Legendre with Condon-Shortley phase (scipy includes it)
        P = lpmv(am, l_val, ct)
        
        if m == 0:
            Y = norm * P
        elif m > 0:
            Y = np.sqrt(2) * norm * P * np.cos(m * ph)
        else:
            Y = np.sqrt(2) * norm * P * np.sin(am * ph)
        
        results[m] = Y
    
    return results

def _compute_f(coeffs, m_vals, Y_dict):
    """Compute f = sum_m c_m * Y_m on the grid."""
    f = np.zeros_like(next(iter(Y_dict.values())))
    for i, m in enumerate(m_vals):
        f += coeffs[i] * Y_dict[m]
    return f

def _compute_ratio(f, weights):
    """Compute ||f||_4 / ||f||_2 using quadrature weights."""
    # weights shape: (n_theta, n_phi) for the full 2D quadrature
    f4 = np.sum(weights * np.abs(f)**4)
    f2 = np.sum(weights * np.abs(f)**2)
    if f2 < 1e-30:
        return np.inf
    return np.sqrt(f4) / np.sqrt(f2)

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Returns 21 coefficients ordered m = -10..10.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    k = 10
    m_vals = np.arange(-k, k + 1)  # -10, -9, ..., 0, ..., 9, 10
    n_m = 2 * k + 1  # 21
    
    # Set up quadrature
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights for cos(theta) in [-1, 1]
    x_gl, w_gl = _gauss_legendre(n_theta)
    
    # phi values
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Build 2D weight matrix
    # The integral over the sphere: ∫∫ f dΩ = ∫_{-1}^{1} ∫_0^{2π} f dφ dμ
    # where μ = cos(θ)
    # So weight for each (θ_i, φ_j) is w_gl[i] * (2π/n_phi)
    dphi = 2 * np.pi / n_phi
    W_1d = w_gl * dphi  # shape (n_theta,)
    W_2d = np.outer(W_1d, np.ones(n_phi))  # shape (n_theta, n_phi)
    
    # Compute spherical harmonics on the grid
    Y_dict = _spherical_harmonics_real(k, m_vals, x_gl, phi_vals)
    
    # Stack Y values into a matrix for efficient computation
    # Y_matrix shape: (n_theta * n_phi, n_m)
    Y_flat = np.zeros((n_theta * n_phi, n_m))
    for i, m in enumerate(m_vals):
        Y_flat[:, i] = Y_dict[m].flatten()
    
    W_flat = W_2d.flatten()  # shape (n_theta * n_phi,)
    
    def _eval(coeffs):
        """Evaluate f and compute ratio."""
        f = Y_flat @ coeffs
        f2 = np.sum(W_flat * f**2)
        if f2 < 1e-30:
            return np.inf, f, f2
        f4 = np.sum(W_flat * f**4)
        ratio = np.sqrt(f4) / np.sqrt(f2)
        return ratio, f, f2
    
    def _grad(coeffs):
        """Compute gradient of ratio w.r.t. coefficients."""
        ratio, f, f2 = _eval(coeffs)
        if not np.isfinite(ratio):
            return np.zeros(n_m)
        
        # d/dc_i (||f||_4 / ||f||_2)
        # = (1/||f||_2) * d||f||_4/dc_i - (||f||_4 / ||f||_2^3) * d||f||_2/dc_i
        # ||f||_4 = (∫f^4)^(1/2), ||f||_2 = (∫f^2)^(1/2)
        # d||f||_4/dc_i = (1/2)(∫f^4)^(-1/2) * 4∫f^3 Y_i = 2(∫f^4)^(-1/2) ∫f^3 Y_i
        # d||f||_2/dc_i = (1/2)(∫f^2)^(-1/2) * 2∫f Y_i = (∫f^2)^(-1/2) ∫f Y_i
        
        f4_int = np.sum(W_flat * f**4)
        f2_int = np.sum(W_flat * f**2)
        
        if f4_int < 1e-30 or f2_int < 1e-30:
            return np.zeros(n_m)
        
        norm4 = np.sqrt(f4_int)
        norm2 = np.sqrt(f2_int)
        
        # ∫f^3 Y_i for each i
        f3Y = Y_flat.T @ (W_flat * f**3)  # shape (n_m,)
        # ∫f Y_i for each i
        fY = Y_flat.T @ (W_flat * f)  # shape (n_m,)
        
        grad = (2.0 / norm4) * f3Y - (norm4 / norm2**3) * fY
        # Simplify: norm4/norm2^3 = (sqrt(f4_int))/norm2^3
        # Actually let me redo this more carefully
        # R = norm4/norm2
        # dR/dc_i = (1/norm2) * d(norm4)/dc_i - (norm4/norm2^2) * d(norm2)/dc_i
        # d(norm4)/dc_i = (1/(2*norm4)) * 4 * ∫f^3 Y_i = (2/norm4) * f3Y[i]
        # d(norm2)/dc_i = (1/(2*norm2)) * 2 * ∫f Y_i = (1/norm2) * fY[i]
        
        grad = (1.0/norm2) * (2.0/norm4) * f3Y - (norm4/norm2**2) * (1.0/norm2) * fY
        grad = (2.0/(norm2*norm4)) * f3Y - (norm4/norm2**3) * fY
        
        return grad
    
    # Initialize with all ones
    coeffs = np.ones(n_m, dtype=np.float64)
    
    # Normalize initial
    _, f_init, f2_init = _eval(coeffs)
    if f2_init > 0:
        coeffs = coeffs / np.sqrt(f2_init)
    
    best_ratio, _, _ = _eval(coeffs)
    best_coeffs = coeffs.copy()
    
    # Gradient descent with adaptive step size
    lr = 0.1
    max_iter = 5000
    patience = 200
    no_improve = 0
    min_improvement = 1e-12
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        ratio, f, f2 = _eval(coeffs)
        if not np.isfinite(ratio):
            break
        
        if ratio < best_ratio - min_improvement:
            best_ratio = ratio
            best_coeffs = coeffs.copy()
            no_improvement = 0
        else:
            no_improvement += 1
        
        if no_improvement > patience:
            # Reduce learning rate
            lr *= 0.5
            no_improvement = 0
            if lr < 1e-10:
                break
        
        grad = _grad(coeffs)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        # Gradient step
        coeffs_new = coeffs - lr * grad / grad_norm * np.linalg.norm(coeffs)
        
        # Check validity
        ratio_new, _, f2_new = _eval(coeffs_new)
        if not np.isfinite(ratio_new) or f2_new < 1e-30:
            lr *= 0.5
            continue
        
        if ratio_new < ratio:
            coeffs = coeffs_new
        else:
            # Backtrack
            lr *= 0.5
            coeffs_new = coeffs - lr * grad / grad_norm * np.linalg.norm(coeffs)
            ratio_new, _, f2_new = _eval(coeffs_new)
            if np.isfinite(ratio_new) and f2_new > 1e-30 and ratio_new < ratio:
                coeffs = coeffs_new
            else:
                # Try smaller step
                for _ in range(10):
                    lr *= 0.5
                    coeffs_new = coeffs - lr * grad / grad_norm * np.linalg.norm(coeffs)
                    ratio_new, _, f2_new = _eval(coeffs_new)
                    if np.isfinite(ratio_new) and f2_new > 1e-30 and ratio_new < ratio:
                        coeffs = coeffs_new
                        break
    
    # Final polish with smaller steps
    if time.time() < deadline - 30:
        lr = 1e-4
        for it in range(2000):
            if time.time() > deadline:
                break
            
            ratio, f, f2 = _eval(coeffs)
            if not np.isfinite(ratio):
                break
            
            if ratio < best_ratio - 1e-14:
                best_ratio = ratio
                best_coeffs = coeffs.copy()
            
            grad = _grad(coeffs)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-16:
                break
            
            coeffs_new = coeffs - lr * grad
            ratio_new, _, f2_new = _eval(coeffs_new)
            if np.isfinite(ratio_new) and f2_new > 1e-30 and ratio_new < ratio:
                coeffs = coeffs_new
            else:
                lr *= 0.5
                if lr < 1e-12:
                    break
    
    # Use best found
    result = best_coeffs.copy()
    
    # Ensure validity: max|c| in [1e-3, 1e5]
    max_c = np.max(np.abs(result))
    if max_c < 1e-3:
        result = result * (1e-3 / max_c)
    elif max_c > 1e5:
        result = result * (1e5 / max_c)
    
    # Final check
    ratio_final, _, f2_final = _eval(result)
    if not np.isfinite(ratio_final) or f2_final < 1e-30:
        # Fall back to all ones
        result = np.ones(n_m, dtype=np.float64)
    
    return result

# EVOLVE-BLOCK-END
