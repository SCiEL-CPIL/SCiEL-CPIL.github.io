# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv, gamma
from scipy.integrate import gauss_legendre


def rudin_problem10() -> np.ndarray:
    """
    Minimize L4/L2 ratio for degree-10 spherical harmonic using conjugate
    gradient on the complex coefficient sphere.
    """
    k = 10
    n_pts_theta = 70
    n_pts_phi = 130
    
    # Gauss-Legendre nodes and weights for cos(theta) in [-1, 1]
    x, w = gauss_legendre(n_pts_theta)
    # theta from arccos(x), dtheta = -dx/sin(theta)
    # But we integrate over dOmega = sin(theta) dtheta dphi = dx dphi (since dx = -sin(theta) dtheta)
    # So ∫∫ g dOmega = ∫_{-1}^{1} ∫_0^{2π} g(x, phi) dphi dx
    # With Gauss-Legendre: ≈ Σ_i w_i ∫_0^{2π} g(x_i, phi) dphi
    # With equally spaced phi: ≈ Σ_i w_i * (2π/n_phi) * Σ_j g(x_i, phi_j)
    
    phi = np.linspace(0, 2 * np.pi, n_pts_phi, endpoint=False)
    dphi = 2 * np.pi / n_pts_phi
    
    # Precompute cos(theta) and sin(theta) for each theta point
    cos_theta = x
    sin_theta = np.sqrt(np.maximum(0, 1 - x**2))
    
    # Precompute Y_10^m for all m and all (theta, phi) grid points
    # Y_l^m = N_l^m P_l^m(cos theta) e^{im phi}
    # N_l^m = sqrt((2l+1)/(4π) * (l-m)!/(l+m)!)
    
    def complex_harmonics(cos_th, sin_th, phi_vals):
        """Compute Y_10^m for m=-10..10 at given points. Returns array of shape (n_m, n_th, n_phi)."""
        n_m = 2 * k + 1
        n_th = len(cos_th)
        n_phi = len(phi_vals)
        Y = np.zeros((n_m, n_th, n_phi), dtype=np.complex128)
        for m_idx in range(n_m):
            m = m_idx - k
            abs_m = abs(m)
            # Normalization
            norm = np.sqrt((2 * k + 1) / (4 * np.pi) * gamma(k - abs_m + 1) / gamma(k + abs_m + 1))
            # Associated Legendre with Condon-Shortley phase
            P = lpmv(abs_m, k, cos_th)  # includes Condon-Shortley phase (-1)^m
            # e^{im phi}
            exp_imphi = np.exp(1j * m * phi_vals)
            Y[m_idx, :, :] = norm * P[:, np.newaxis] * exp_imphi[np.newaxis, :]
        return Y
    
    Y_grid = complex_harmonics(cos_theta, sin_theta, phi)
    # Y_grid shape: (21, 70, 130)
    
    # Quadrature weights: w_i * dphi for each (i, j)
    # Total weight for point (i,j) is w_i * dphi
    quad_weights = w[:, np.newaxis] * dphi  # shape (70, 1)
    # Full weight array: (70, 130)
    W = np.tile(quad_weights, (1, n_pts_phi))
    
    def compute_f(a):
        """Compute f on the grid given complex coefficients a (shape 21)."""
        # f = sum_m a_m Y_m
        # a shape (21,), Y_grid shape (21, 70, 130)
        f = np.tensordot(a, Y_grid, axes=(0, 0))
        return f  # shape (70, 130)
    
    def compute_l4_norm(a):
        """Compute ||f||_4 = (∫|f|^4 dOmega)^{1/4}."""
        f = compute_f(a)
        integral = np.sum(np.abs(f)**4 * W)
        return integral**0.25
    
    def compute_l2_norm(a):
        """Compute ||f||_2 = (∫|f|^2 dOmega)^{1/2} = (sum |a_m|^2)^{1/2} by orthonormality."""
        return np.sqrt(np.sum(np.abs(a)**2))
    
    def compute_ratio(a):
        """Compute R = ||f||_4 / ||f||_2."""
        l4 = compute_l4_norm(a)
        l2 = compute_l2_norm(a)
        if l2 < 1e-15:
            return np.inf
        return l4 / l2
    
    def compute_gradient(a):
        """Compute gradient of ||f||_4 with respect to a (conjugate).
        ∂||f||_4/∂a_k* = (1/||f||_4^3) ∫ |f|^2 f Y_k* dOmega
        """
        f = compute_f(a)
        l4 = compute_l4_norm(a)
        if l4 < 1e-15:
            return np.zeros(2 * k + 1, dtype=np.complex128)
        
        # integrand = |f|^2 f Y_k* for each k
        # grad_k = (1/l4^3) * sum_{i,j} |f_ij|^2 f_ij Y_k_ij* W_ij
        abs_f_sq = np.abs(f)**2  # shape (70, 130)
        integrand_base = abs_f_sq * f  # shape (70, 130)
        
        grad = np.zeros(2 * k + 1, dtype=np.complex128)
        for m_idx in range(2 * k + 1):
            # Y_grid[m_idx] shape (70, 130)
            integral = np.sum(integrand_base * np.conj(Y_grid[m_idx]) * W)
            grad[m_idx] = integral / (l4**3)
        
        return grad
    
    # Initialize: try a random point on the complex sphere
    np.random.seed(42)
    a = np.random.randn(2 * k + 1) + 1j * np.random.randn(2 * k + 1)
    a = a / np.linalg.norm(a)
    
    # Conjugate gradient on the sphere
    best_a = a.copy()
    best_ratio = compute_ratio(a)
    
    max_iter = 200
    for it in range(max_iter):
        # Compute gradient
        g = compute_gradient(a)
        
        # Project gradient onto tangent space: g_tan = g - (g · a) a
        # where g · a = sum conj(g_k) a_k
        g_dot_a = np.vdot(g, a)  # sum conj(g_k) a_k
        g_tan = g - g_dot_a * a
        
        # Check convergence
        g_tan_norm = np.linalg.norm(g_tan)
        if g_tan_norm < 1e-10:
            break
        
        # Line search: minimize ratio along a + alpha * g_tan (then renormalize)
        # Since ratio is scale-invariant, we can fix ||a||_2 = 1
        # Try a few alpha values
        best_alpha = 0.0
        best_val = compute_ratio(a)
        
        # Use backtracking line search
        alpha = 1.0
        for ls_iter in range(30):
            a_new = a + alpha * g_tan
            a_new = a_new / np.linalg.norm(a_new)
            val = compute_ratio(a_new)
            if val < best_val:
                best_val = val
                best_alpha = alpha
                best_a_new = a_new
            alpha *= 0.5
            if alpha < 1e-12:
                break
        
        if best_alpha == 0.0:
            break
        
        a = best_a_new
        if compute_ratio(a) < best_ratio:
            best_ratio = compute_ratio(a)
            best_a = a.copy()
    
    # Convert complex coefficients to real coefficients
    # Real harmonics: Y_0_real = Y_10^0 (m=0 is real)
    # For m > 0: Y_m_real = (Y_10^m + Y_10^{-m}) / sqrt(2)
    # For m < 0: Y_m_real = (Y_10^m - Y_10^{-m}) / (i * sqrt(2)) = (Y_10^{-|m|} - Y_10^{|m|}) / (i * sqrt(2))
    
    # f = sum_{m=-10}^{10} a_m Y_10^m
    # Also f = sum_{m=-10}^{10} c_m Y_m_real
    
    # Let's express: for m > 0, Y_m_real = (Y^m + Y^{-m})/sqrt(2)
    # Y^{-m} = conj(Y^m) since Y^{-m} = N P e^{-im phi} and Y^m = N P e^{im phi}, P real
    # So Y_m_real = (Y^m + conj(Y^m))/sqrt(2) = sqrt(2) Re(Y^m)
    # Y_{-m}_real = (Y^m - Y^{-m})/(i sqrt(2)) = (Y^m - conj(Y^m))/(i sqrt(2)) = sqrt(2) Im(Y^m)
    
    # So f = a_0 Y^0 + sum_{m=1}^{10} a_m Y^m + sum_{m=1}^{10} a_{-m} Y^{-m}
    # = a_0 Y^0 + sum_{m=1}^{10} [a_m Y^m + a_{-m} conj(Y^m)]
    # = a_0 Y^0 + sum_{m=1}^{10} [(a_m + a_{-m}) Re(Y^m) + i(a_m - a_{-m}) Im(Y^m)]
    
    # And f = c_0 Y_0_real + sum_{m=1}^{10} c_m Y_m_real + sum_{m=1}^{10} c_{-m} Y_{-m}_real
    # = c_0 Y^0 + sum_{m=1}^{10} [c_m sqrt(2) Re(Y^m) + c_{-m} sqrt(2) Im(Y^m)]
    
    # Matching:
    # c_0 = a_0
    # c_m sqrt(2) = a_m + a_{-m}  => c_m = (a_m + a_{-m})/sqrt(2)
    # c_{-m} sqrt(2) = i(a_m - a_{-m}) => c_{-m} = i(a_m - a_{-m})/sqrt(2)
    
    coeffs = np.zeros(2 * k + 1, dtype=np.float64)
    for m_idx in range(2 * k + 1):
        m = m_idx - k
        if m == 0:
            coeffs[m_idx] = np.real(a[m_idx])
        elif m > 0:
            coeffs[m_idx] = np.real((a[m_idx] + a[2 * k - m_idx]) / np.sqrt(2))
        else:  # m < 0
            abs_m = -m
            # c_{-m} = i(a_m - a_{-m})/sqrt(2) where m here is the index for negative
            # m_idx corresponds to m (negative), -m = abs_m, index of -m is k + abs_m = k - m
            idx_pos = k + abs_m  # index for +abs_m
            idx_neg = m_idx       # index for m (negative)
            coeffs[m_idx] = np.real(1j * (a[idx_neg] - a[idx_pos]) / np.sqrt(2))
    
    # Ensure validity
    if not np.all(np.isfinite(coeffs)):
        coeffs = np.ones(2 * k + 1, dtype=np.float64)
    max_abs = np.max(np.abs(coeffs))
    if max_abs < 1e-3:
        coeffs = coeffs / max_abs if max_abs > 0 else np.ones(2 * k + 1)
    elif max_abs > 1e5:
        coeffs = coeffs / max_abs * 1e5
    
    return coeffs


# EVOLVE-BLOCK-END

