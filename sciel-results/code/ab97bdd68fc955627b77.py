import numpy as np
from scipy.special import lpmv
import math

def _precompute():
    l = 10
    N_theta = 70
    N_phi = 130
    
    x, w_gl = np.polynomial.legendre.leggauss(N_theta)
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    
    Phi, X = np.meshgrid(phi, x, indexing='ij')
    weights = np.outer(np.ones(N_phi), w_gl) * (2 * np.pi / N_phi)
    
    Y = np.zeros((N_phi, N_theta, 21), dtype=np.float64)
    
    for m_idx, m in enumerate(range(-10, 11)):
        abs_m = abs(m)
        log_N = 0.5 * (np.log(21.0) - np.log(4.0 * np.pi) + math.lgamma(11 - abs_m) - math.lgamma(11 + abs_m))
        N_m = np.exp(log_N)
        P_lm = lpmv(abs_m, l, X)
        
        if m == 0:
            Y[:, :, m_idx] = N_m * P_lm
        elif m > 0:
            Y[:, :, m_idx] = np.sqrt(2.0) * N_m * P_lm * np.cos(m * Phi)
        else:
            Y[:, :, m_idx] = np.sqrt(2.0) * N_m * P_lm * np.sin(abs_m * Phi)
    
    Y_flat = Y.flatten()  # (N_points, 21)
    w_flat = weights.flatten()
    return Y_flat, w_flat

Y_mat, w_vec = _precompute()
N_pts, N_coeff = Y_mat.shape

def _objective_and_grad(c):
    """
    Compute L4/L2 ratio and its gradient w.r.t. c, assuming ||c||=1.
    Returns (value, grad) where grad is the Euclidean gradient.
    """
    f = Y_mat @ c  # (N_pts,)
    f4 = f ** 4
    integral_l4 = np.dot(w_vec, f4)
    l4_norm = integral_l4 ** 0.25
    
    # L2 norm = 1 (constrained)
    # d/dc [l4_norm] = (1/4) * integral_l4^(-3/4) * 4 * sum w_i * f_i^3 * Y_i
    # = integral_l4^(-3/4) * sum w_i * f_i^3 * Y_i
    
    if integral_l4 <= 0:
        return 1e10, np.zeros(N_coeff)
    
    coeff_grad = integral_l4 ** (-0.75) * np.dot(w_vec * f**3, Y_mat.T)
    # Wait, let me redo this properly
    # l4_norm = (sum w_i f_i^4)^(1/4)
    # dl4/dc_j = (1/4)(sum w_i f_i^4)^(-3/4) * sum_i w_i * 4 f_i^3 * Y_ij
    #          = (sum w_i f_i^4)^(-3/4) * sum_i w_i f_i^3 Y_ij
    
    grad = integral_l4 ** (-0.75) * (w_vec * f**3) @ Y_mat
    
    return l4_norm, grad

def _riemannian_cg(c0, max_iter=200, tol=1e-12):
    """
    Riemannian conjugate gradient on the unit sphere.
    """
    c = c0 / np.linalg.norm(c0)
    val, grad = _objective_and_grad(c)
    
    # Project gradient to tangent space
    grad_tan = grad - np.dot(grad, c) * c
    d = -grad_tan
    d_norm = np.linalg.norm(d)
    if d_norm < tol:
        return c, val
    
    d = d / d_norm
    alpha = 1.0
    
    for it in range(max_iter):
        # Line search
        f_new = None
        for k in range(20):
            c_trial = c + alpha * d
            c_trial = c_trial / np.linalg.norm(c_trial)
            val_trial, grad_trial = _objective_and_grad(c_trial)
            if val_trial < val:
                f_new = val_trial
                grad_trial_new = grad_trial
                break
            alpha *= 0.5
        else:
            break
        
        if f_new is None:
            break
        
        # Check convergence
        grad_trial_tan = grad_trial_new - np.dot(grad_trial_new, c_trial) * c_trial
        if np.linalg.norm(grad_trial_tan) < tol:
            c = c_trial
            val = val_trial
            break
        
        # Conjugate gradient update
        beta = np.dot(grad_trial_tan, grad_trial_tan) / max(np.dot(grad_tan, grad_tan), 1e-15)
        d = -grad_trial_tan + beta * d
        d_tan = d - np.dot(d, c_trial) * c_trial
        d_norm = np.linalg.norm(d_tan)
        if d_norm < tol:
            c = c_trial
            val = val_trial
            break
        d = d_tan / d_norm
        c = c_trial
        val = val_trial
        grad_tan = grad_trial_tan
        alpha = 1.0
    
    return c, val

def rudin_problem10() -> np.ndarray:
    n_coeffs = 21
    rng = np.random.default_rng(12345)
    
    best_c = None
    best_val = np.inf
    
    n_restarts = 100
    
    for i in range(n_restarts):
        c0 = rng.normal(size=n_coeffs)
        c0 /= np.linalg.norm(c0)
        
        try:
            c_opt, val_opt = _riemannian_cg(c0, max_iter=300, tol=1e-12)
            if val_opt < best_val:
                best_val = val_opt
                best_c = c_opt
        except Exception:
            pass
    
    if best_c is None:
        return np.ones(n_coeffs, dtype=np.float64)
    
    return best_c.astype(np.float64)
