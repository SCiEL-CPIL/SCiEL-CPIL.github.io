# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, gammaln
from scipy.optimize import minimize

_DEADLINE = time.time() + 610

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
    Compute the matrix X where X[i, j] = Y_j(ct_i, phi_j).
    j ranges from 0 to 20, corresponding to m = -10 to 10.
    """
    N_pts_ct = len(cos_theta)
    N_pts_phi = len(phi)
    N_pts = N_pts_ct * N_pts_phi
    X = np.zeros((N_pts, 21))
    
    ct, ph = np.meshgrid(cos_theta, phi, indexing='ij')
    ct_flat = ct.flatten()
    ph_flat = ph.flatten()
    
    for k, m in enumerate(range(-10, 11)):
        abs_m = abs(m)
        log_N = 0.5 * (np.log(21.0 / (4.0 * np.pi)) + 
                       gammaln(10 - abs_m + 1) - gammaln(10 + abs_m + 1))
        N_m = np.exp(log_N)
        
        P = lpmv(abs_m, 10, ct_flat)
        
        if m == 0:
            Y = N_m * P
        elif m > 0:
            Y = np.sqrt(2.0) * N_m * P * np.cos(m * ph_flat)
        else:
            Y = np.sqrt(2.0) * N_m * P * np.sin(abs_m * ph_flat)
            
        X[:, k] = Y
        
    return X

def _ratio_and_grad(c, X, w_flat):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient in one pass.
    """
    f = X @ c
    w_f = w_flat * f
    w_f2 = w_flat * f * f
    w_f3 = w_f2 * f
    w_f4 = w_f2 * w_f2
    
    S2 = np.sum(w_f2)
    S4 = np.sum(w_f4)
    
    if S2 < 1e-10 or S4 < 1e-10:
        return 1e10, np.zeros(21)
        
    N2 = np.sqrt(S2)
    N4 = S4 ** 0.25
    R = N4 / N2
    
    term1 = X.T @ w_f3
    term2 = X.T @ w_f
    
    # Correct gradient: d/dR = 1/4 * dS4/S4 - 1/2 * dS2/S2
    grad_log_R = 0.25 * (term1 / S4) - 0.5 * (term2 / S2)
    grad_R = R * grad_log_R
    
    return R, grad_R

def _optimize_on_sphere(c0, X, w_flat, max_iter=2000, tol=1e-12):
    """
    Gradient descent on the unit sphere with backtracking line search.
    """
    c = c0 / np.linalg.norm(c0)
    prev_val = np.inf
    
    for i in range(max_iter):
        if time.time() > _DEADLINE - 15:
            break
            
        val, grad = _ratio_and_grad(c, X, w_flat)
        if not np.isfinite(val):
            break
            
        if abs(prev_val - val) < tol:
            break
        prev_val = val
        
        # Project gradient to tangent space
        g_t = grad - np.dot(grad, c) * c
        g_norm = np.linalg.norm(g_t)
        
        if g_norm < 1e-10:
            break
            
        # Backtracking line search
        alpha = 1.0
        c_new = c
        val_new = val
        
        for _ in range(30):
            c_trial = c - alpha * g_t / g_norm
            norm_trial = np.linalg.norm(c_trial)
            if norm_trial < 1e-10:
                alpha *= 0.5
                continue
            c_trial = c_trial / norm_trial
            
            val_trial, _ = _ratio_and_grad(c_trial, X, w_flat)
            if val_trial < val:
                c_new = c_trial
                val_new = val_trial
                break
            alpha *= 0.5
            
        if not (val_new < val):
            # If no decrease found, we are likely at a local minimum or stuck
            break
            
        c = c_new

    return c, val

def rudin_problem10() -> np.ndarray:
    """
    Minimize ||f||_4 / ||f||_2 for degree-10 real spherical harmonics.
    Returns 21 coefficients c_m for m = -10..10.
    """
    global _DEADLINE
    
    cos_theta, phi, w_1d = _build_quadrature()
    X = _compute_harmonics_matrix(cos_theta, phi)
    w_flat = np.tile(w_1d, 130)
    
    best_coeffs = None
    best_val = np.inf
    
    rng = np.random.default_rng(42)
    
    starts = []
    # m=0
    c0 = np.zeros(21)
    c0[10] = 1.0
    starts.append(c0)
    
    # m=10
    c10 = np.zeros(21)
    c10[20] = 1.0
    starts.append(c10)
    
    # m=-10
    c_10 = np.zeros(21)
    c_10[0] = 1.0
    starts.append(c_10)
    
    # m=5, -5
    c5 = np.zeros(21)
    c5[15] = 1.0
    starts.append(c5)
    
    c_5 = np.zeros(21)
    c_5[5] = 1.0
    starts.append(c_5)
    
    # m=4, -4
    c4 = np.zeros(21)
    c4[14] = 1.0
    starts.append(c4)
    
    c_4 = np.zeros(21)
    c_4[6] = 1.0
    starts.append(c_4)
    
    # m=8, -8
    c8 = np.zeros(21)
    c8[18] = 1.0
    starts.append(c8)
    
    c_8 = np.zeros(21)
    c_8[2] = 1.0
    starts.append(c_8)
    
    # m=2, -2
    c2 = np.zeros(21)
    c2[12] = 1.0
    starts.append(c2)
    
    c_2 = np.zeros(21)
    c_2[8] = 1.0
    starts.append(c_2)
    
    # Structured multi-m starts: pairs of m values
    m_pairs = [(0,10),(0,8),(0,6),(0,4),(0,2),
               (10,8),(10,6),(10,4),(10,2),
               (8,6),(8,4),(8,2),(6,4),(6,2),(4,2),
               (0,-10),(0,-8),(0,-6),(0,-4),(0,-2),
               (10,-10),(8,-8),(6,-6),(4,-4),(2,-2)]
    for m1, m2 in m_pairs:
        c = np.zeros(21)
        c[m1 + 10] = 1.0 / np.sqrt(2.0)
        c[m2 + 10] = 1.0 / np.sqrt(2.0)
        starts.append(c)
    
    # Triple m starts
    m_triples = [(0, 10, 5), (0, 8, 4), (0, 6, 3), (0, 4, 2),
                 (10, 8, 6), (10, 6, 4), (10, 4, 2),
                 (0, -10, 5), (0, -8, 4), (0, -6, 3)]
    for m1, m2, m3 in m_triples:
        c = np.zeros(21)
        c[m1 + 10] = 1.0 / np.sqrt(3.0)
        c[m2 + 10] = 1.0 / np.sqrt(3.0)
        c[m3 + 10] = 1.0 / np.sqrt(3.0)
        starts.append(c)
    
    # Random starts
    for _ in range(300):
        c = rng.standard_normal(21)
        c = c / np.linalg.norm(c)
        starts.append(c)
        
    for start in starts:
        if time.time() > _DEADLINE - 15:
            break
            
        try:
            res_c, val = _optimize_on_sphere(start, X, w_flat)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = res_c.copy()
        except Exception:
            pass

    # Refine best result with more iterations if time permits
    if best_coeffs is not None and time.time() < _DEADLINE - 30:
        try:
            res_c, val = _optimize_on_sphere(best_coeffs, X, w_flat, max_iter=5000, tol=1e-14)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = res_c.copy()
        except Exception:
            pass
    
    # Additional polish pass with even tighter tolerance
    if best_coeffs is not None and time.time() < _DEADLINE - 10:
        try:
            res_c, val = _optimize_on_sphere(best_coeffs, X, w_flat, max_iter=10000, tol=1e-15)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = res_c.copy()
        except Exception:
            pass

    if best_coeffs is None:
        best_coeffs = np.zeros(21)
        best_coeffs[10] = 1.0

    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs * (0.01 / max_abs)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs * (1e4 / max_abs)
        
    return best_coeffs

# EVOLVE-BLOCK-END
