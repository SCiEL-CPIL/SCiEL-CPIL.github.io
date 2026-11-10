# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

def rudin_problem10() -> np.ndarray:
    k = 10
    n_coeffs = 2 * k + 1
    
    # Deadline guard
    t_start = time.time()
    deadline = 620.0  # 10s margin
    
    # Quadrature setup
    # 70 Gauss-Legendre points in cos(theta)
    x, w_theta = np.polynomial.legendre.leggauss(70)
    # Map to theta? No, x is cos(theta).
    # dOmega = sin(theta) dtheta dphi = -dcos(theta) dphi
    # Integral over theta from 0 to pi of g(cos(theta)) dcos(theta) is integral -1 to 1.
    # So weight for cos(theta) grid is just w_theta.
    # Total weight sum should be 2 for theta integral?
    # sum(w_theta) = 2.
    # Integral over sphere is 1/2 * sum_theta w_theta * sum_phi (2pi/N_phi) ...
    # Actually, standard Gauss-Legendre on [-1,1] approximates \int_{-1}^1 g(x) dx.
    # \int_0^\pi g(\cos \theta) \sin \theta d\theta = \int_{-1}^1 g(x) dx.
    # So \int_{S^2} f d\Omega = \int_0^{2\pi} d\phi \int_{-1}^1 f(x, \phi) dx.
    # Discrete: \sum_{i=0}^{N_phi-1} \frac{2\pi}{N_phi} \sum_{j=0}^{N_theta-1} w_{\theta, j} f(x_j, \phi_i).
    
    N_phi = 130
    phi = np.arange(N_phi) * (2 * np.pi / N_phi)
    
    # Weights for the full sphere integral
    # w_sphere[i, j] = (2*pi/N_phi) * w_theta[j]
    # But we can separate them for efficiency.
    w_sphere = w_theta[:, None] * (2 * np.pi / N_phi)  # Shape (70, 130)
    
    # Precompute Basis Functions Y_m(x_j, phi_i)
    # m ranges -10 to 10
    # Y_0 = N_0 P^0
    # Y_m (m>0) = sqrt(2) N_m P^m cos(m phi)
    # Y_m (m<0) = sqrt(2) N_|m| P^|m| sin(|m| phi)
    
    # P_l^m(x) with Condon-Shortley phase
    # scipy.special.lpmv(m, l, x) includes CS phase.
    
    Y_vals = np.zeros((n_coeffs, 70, 130), dtype=np.float64)
    
    for idx, m in enumerate(range(-k, k + 1)):
        m_abs = abs(m)
        # Normalization N_m
        # N_m = sqrt( (2k+1)/(4*pi) * (k-m_abs)! / (k+m_abs)! )
        # Using factorial or gammaln for stability? k=10 is small.
        num = (k - m_abs)
        den = (k + m_abs)
        # Factorials
        # (k-m)! / (k+m)!
        # Can compute directly
        ratio = 1.0
        if k + m_abs != k - m_abs:
            # Product from k-m_abs+1 to k+m_abs
            for i in range(k - m_abs + 1, k + m_abs + 1):
                ratio /= i
        norm_const = np.sqrt((2 * k + 1) / (4 * np.pi) * ratio)
        
        # Associated Legendre
        P = lpmv(m_abs, k, x) # Shape (70,)
        
        if m == 0:
            # Y_0
            Y_vals[idx, :, :] = norm_const * P[:, None] * np.ones((1, 130))
        elif m > 0:
            # Y_m
            cos_m_phi = np.cos(m * phi) # Shape (130,)
            Y_vals[idx, :, :] = (np.sqrt(2) * norm_const * P[:, None]) * cos_m_phi[None, :]
        else:
            # Y_m (m < 0)
            sin_m_phi = np.sin(m_abs * phi) # Shape (130,)
            Y_vals[idx, :, :] = (np.sqrt(2) * norm_const * P[:, None]) * sin_m_phi[None, :]
            
    # Flatten Y_vals for faster dot products?
    # Y_flat shape (21, 9100)
    Y_flat = Y_vals.reshape(n_coeffs, -1)
    w_flat = w_sphere.flatten()
    
    # Objective and Gradient
    def objective(c):
        f = Y_flat @ c # Shape (9100,)
        # U = ||f||_4^4 = sum w_i f_i^4
        U = np.dot(w_flat, f**4)
        # V = ||f||_2^2 = sum c_i^2 (exact due to orthonormality on grid)
        V = np.dot(c, c)
        if V == 0:
            return 1e10, np.zeros_like(c)
        val = U / (V**2)
        # Gradient
        # grad U_k = 4 * sum w_i f_i^3 Y_k(x_i) = 4 * <f^3, Y_k>
        # grad V_k = 2 c_k
        # grad F = (grad U * V^2 - U * 2 * V * grad V) / V^4
        #         = (grad U) / V^2 - 2 * U * grad V / V^3
        #         = (grad U) / V^2 - 4 * U * c / V^3
        
        f_cubed = f**3
        grad_U = 4.0 * (Y_flat.T @ (w_flat * f_cubed)) # Shape (21,)
        
        grad_F = grad_U / (V**2) - (4.0 * U / (V**3)) * c
        
        return val, grad_F

    best_c = np.ones(n_coeffs)
    best_score = np.inf
    
    n_restarts = 500
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Random initial guess
        # Mix of random and structured
        if i % 2 == 0:
            c0 = np.random.randn(n_coeffs)
        else:
            # Try emphasizing high m or low m
            m_target = np.random.randint(0, k + 1)
            c0 = np.zeros(n_coeffs)
            # index for m_target
            idx_pos = m_target + k
            idx_neg = -m_target + k
            if m_target == 0:
                c0[idx_pos] = 1.0
            else:
                # Random combination of cos and sin
                c0[idx_pos] = np.random.randn()
                c0[idx_neg] = np.random.randn()
                
        # Normalize
        norm_c0 = np.linalg.norm(c0)
        if norm_c0 < 1e-6:
            c0 = np.ones(n_coeffs) / np.sqrt(n_coeffs)
        else:
            c0 = c0 / norm_c0

        try:
            res = minimize(objective, c0, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success or res.status == 0:
                c_opt = res.x
                score, _ = objective(c_opt)
                if score < best_score:
                    best_score = score
                    best_c = c_opt
        except Exception:
            pass

    # Final cleanup
    # Ensure valid size and range
    if not isinstance(best_c, np.ndarray) or best_c.shape != (n_coeffs,):
        return np.ones(n_coeffs)
        
    # Check max|c| constraints
    max_c = np.max(np.abs(best_c))
    if max_c < 1e-3:
        best_c = best_c * (1e-2 / max_c)
    if max_c > 1e5:
        best_c = best_c * (1e4 / max_c)
        
    return best_c

# EVOLVE-BLOCK-END
