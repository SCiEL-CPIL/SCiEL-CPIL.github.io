# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize

def _build_basis(k=10, n_theta=70, n_phi=130):
    """Precompute the spherical harmonic basis matrix Y_flat of shape (n_theta*n_phi, 2*k+1).
    Also returns weights for the quadrature."""
    # Gauss-Legendre quadrature in x = cos(theta)
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # x in [-1, 1], w are weights
    # theta = arccos(x), dOmega = sin(theta) dtheta dphi = -dx dphi
    # So integral over sphere = sum_x sum_phi f(x, phi) * w_x * (2*pi/n_phi)
    
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    dphi = 2*np.pi / n_phi
    
    # Build grid
    X, Phi = np.meshgrid(x, phi, indexing='ij')  # shape (n_theta, n_phi)
    X_flat = X.flatten()  # shape (n_theta*n_phi,)
    Phi_flat = Phi.flatten()
    
    # Weights: w_x * dphi
    w_x_expanded = np.repeat(w, n_phi)  # shape (n_theta*n_phi,)
    weights = w_x_expanded * dphi  # shape (n_theta*n_phi,)
    
    # Total mass should be 4*pi
    # sum(weights) should be 4*pi (since sum(w)=2, times 2*pi/n_phi * n_phi = 4*pi)
    
    # Compute spherical harmonics
    # Y_0 = N_0 P_k^0(cos theta)
    # Y_m for m > 0: sqrt(2) N_m P_k^m(cos theta) cos(m phi)
    # Y_m for m < 0: sqrt(2) N_|m| P_k^|m|(cos theta) sin(|m| phi)
    
    n_points = n_theta * n_phi
    Y_flat = np.zeros((n_points, 2*k+1), dtype=np.float64)
    
    for m in range(-k, k+1):
        abs_m = abs(m)
        # N_m = sqrt((2k+1)/(4*pi) * (k-abs_m)!/(k+abs_m)!)
        N_m = np.sqrt((2*k+1)/(4*np.pi) * np.math.factorial(k-abs_m) / np.math.factorial(k+abs_m))
        
        # P_k^m(cos theta) with Condon-Shortley phase
        # scipy.special.lpmv(m, l, x) gives P_l^m(x) with Condon-Shortley phase
        P_lm = lpmv(abs_m, k, X_flat)  # shape (n_points,)
        
        if m == 0:
            Y_flat[:, m+k] = N_m * P_lm
        elif m > 0:
            Y_flat[:, m+k] = np.sqrt(2) * N_m * P_lm * np.cos(m * Phi_flat)
        else:
            Y_flat[:, m+k] = np.sqrt(2) * N_m * P_lm * np.sin(abs_m * Phi_flat)
    
    return Y_flat, weights

def _compute_R(c, Y_flat, weights):
    """Compute R = ||f||_4 / ||f||_2 where f = Y_flat @ c."""
    f = Y_flat @ c
    norm2_sq = np.sum(weights * f**2)
    norm4_sq = np.sum(weights * f**4)
    if norm2_sq <= 0:
        return np.inf
    return np.sqrt(norm4_sq) / norm2_sq

def _objective_and_grad(c, Y_flat, weights):
    """Compute objective R and its gradient."""
    f = Y_flat @ c
    f2 = f**2
    f4 = f2**2
    
    norm2_sq = np.sum(weights * f2)
    norm4_sq = np.sum(weights * f4)
    
    if norm2_sq <= 0:
        return np.inf, np.zeros_like(c)
    
    # R = sqrt(norm4_sq) / norm2_sq
    # Let A = norm4_sq, B = norm2_sq
    # R = A^0.5 / B
    # dR/dc_i = (0.5 * A^(-0.5) * dA/dc_i * B - A^0.5 * dB/dc_i) / B^2
    #        = (0.5 * A^(-0.5) * B - A^0.5 * B^(-1)) * dA/dc_i / B - A^0.5 * dB/dc_i / B^2
    # Let's simplify:
    # dA/dc_i = 4 * sum(weights * f3 * Y_i)  where f3 = f^3
    # dB/dc_i = 2 * sum(weights * f * Y_i)
    
    # Actually, let me redo this more carefully.
    # R = (sum w f^4)^0.5 / (sum w f^2)
    # Let S4 = sum w f^4, S2 = sum w f^2
    # R = S4^0.5 / S2
    # dR/dc = (0.5 * S4^(-0.5) * dS4/dc * S2 - S4^0.5 * dS2/dc) / S2^2
    #       = S4^(-0.5) * dS4/dc / S2 - S4^0.5 * dS2/dc / S2^2
    
    # dS4/dc_i = 4 * sum_j w_j f_j^3 Y_{j,i}
    # dS2/dc_i = 2 * sum_j w_j f_j Y_{j,i}
    
    w_f3 = weights * f**3  # shape (n_points,)
    w_f = weights * f      # shape (n_points,)
    
    dS4_dc = 4.0 * (Y_flat.T @ w_f3)  # shape (2*k+1,)
    dS2_dc = 2.0 * (Y_flat.T @ w_f)   # shape (2*k+1,)
    
    R = np.sqrt(S4_sq) / S2_sq
    grad = (S4_sq ** (-0.5) * dS4_dc / S2_sq - np.sqrt(S4_sq) * dS2_dc / S2_sq**2)
    
    return R, grad

def _run_l_bfgs(c0, Y_flat, weights, max_iter=1000, time_limit=500):
    """Run L-BFGS-B from initial point c0."""
    def obj_grad(c):
        val, grad = _objective_and_grad(c, Y_flat, weights)
        return val, grad
    
    try:
        res = minimize(obj_grad, c0, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12,
                                'maxls': 50, 'disp': False})
        return res.x, res.fun
    except Exception:
        return c0, _compute_R(c0, Y_flat, weights)

def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 that minimize
    R = ||f||_4 / ||f||_2 on the unit sphere, where f is a degree-10
    spherical harmonic.
    """
    t_start = time.time()
    deadline = t_start + 600  # safety margin
    
    k = 10
    Y_flat, weights = _build_basis(k=k, n_theta=70, n_phi=130)
    n_coeffs = 2*k + 1
    
    # The objective is homogeneous of degree 0 in c (scaling c doesn't change R),
    # so we can fix the norm of c to avoid the flat direction.
    # We'll constrain ||c||_2 = 1 by projecting or just work in that space.
    # Actually, L-BFGS-B doesn't handle equality constraints well.
    # Instead, we can normalize c after each iteration, but that complicates the gradient.
    # Alternative: just run L-BFGS-B and let it handle the scale. The gradient
    # should drive the scale to a stable point or we can normalize at the end.
    # Actually, since R is scale-invariant, the gradient is orthogonal to c
    # (Euler's theorem). So L-BFGS-B should be fine; it will just find a point
    # on the ray.
    
    # Multiple restarts with different initial points
    best_c = None
    best_R = np.inf
    
    rng = np.random.default_rng(42)
    
    # Try several random initial points
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() > deadline - 60:  # leave 60s for polish
            break
        
        # Random initial point
        if i == 0:
            c0 = np.ones(n_coeffs, dtype=np.float64)
        elif i == 1:
            # Only m=0
            c0 = np.zeros(n_coeffs, dtype=np.float64)
            c0[k] = 1.0
        elif i == 2:
            # Only |m|=1
            c0 = np.zeros(n_coeffs, dtype=np.float64)
            c0[k-1] = 1.0
            c0[k+1] = 1.0
        else:
            c0 = rng.standard_normal(n_coeffs)
        
        # Normalize to unit norm to help optimization
        c0 = c0 / np.linalg.norm(c0)
        
        c_opt, R_opt = _run_l_bfgs(c0, Y_flat, weights, max_iter=500, time_limit=300)
        
        if R_opt < best_R:
            best_R = R_opt
            best_c = c_opt.copy()
        
        # Check time
        if time.time() > deadline - 120:
            break
    
    # Final polish with more iterations
    if best_c is not None and time.time() < deadline - 30:
        c_polished, R_polished = _run_l_bfgs(best_c, Y_flat, weights, max_iter=2000, time_limit=200)
        if R_polished < best_R:
            best_R = R_polished
            best_c = c_polished
    
    if best_c is None:
        best_c = np.ones(n_coeffs, dtype=np.float64)
    
    # Validate
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        best_c = np.ones(n_coeffs, dtype=np.float64)
    
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c / max_abs * 1.0
    elif max_abs > 1e5:
        best_c = best_c / max_abs * 1.0
    
    return best_c.astype(np.float64)

# EVOLVE-BLOCK-END
