import numpy as np
import time
import math
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import sys

def rudin_problem10() -> np.ndarray:
    """
    Finds the coefficients c_m for m = -10..10 of a real degree-10 spherical harmonic
    that minimizes the ratio R = ||f||_4 / ||f||_2.
    
    Strategy:
    1. Precompute quadrature nodes and weights, and the full basis matrix Y.
    2. Derive analytic gradients for the objective R = ||f||_4 / ||f||_2
       using chain rule and precomputed basis products.
    3. Use L-BFGS-B with multiple random restarts on the unit sphere.
    4. Time-guarded to return best result before 630s.
    """
    
    k = 10
    n_coeffs = 2 * k + 1
    
    # --- Precomputation ---
    
    # 1. Gauss-Legendre nodes and weights for cos(theta)
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    
    # 2. Equally spaced phi points
    n_phi = 130
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # 3. Spherical Harmonics Basis
    N_m = np.zeros(n_coeffs)
    for m in range(-k, k + 1):
        abs_m = abs(m)
        N_m[m + k] = math.sqrt((2 * k + 1) / (4 * np.pi) * factorial(k - abs_m) / factorial(k + abs_m))
        
    P_vals = np.zeros((len(x_gl), k + 1))
    for m in range(k + 1):
        P_vals[:, m] = lpmv(m, k, x_gl)
        
    cos_m_phi = np.zeros((n_phi, k + 1))
    sin_m_phi = np.zeros((n_phi, k + 1))
    for m in range(k + 1):
        cos_m_phi[:, m] = np.cos(m * phi_vals)
        sin_m_phi[:, m] = np.sin(m * phi_vals)
        
    # Build the basis matrix Y
    # Y_basis shape: (n_u, n_phi, n_coeffs)
    Y_basis = np.zeros((len(x_gl), n_phi, n_coeffs))
    
    for c_idx, m in enumerate(range(-k, k + 1)):
        abs_m = abs(m)
        if m == 0:
            val = N_m[0] * P_vals[:, 0]
            Y_basis[:, :, c_idx] = val[:, np.newaxis] * np.ones((1, n_phi))
        elif m > 0:
            val_u = N_m[m] * P_vals[:, m]
            val_phi = cos_m_phi[:, m]
            Y_basis[:, :, c_idx] = math.sqrt(2) * (val_u[:, np.newaxis] * val_phi[np.newaxis, :])
        else:
            val_u = N_m[abs_m] * P_vals[:, abs_m]
            val_phi = sin_m_phi[:, abs_m]
            Y_basis[:, :, c_idx] = math.sqrt(2) * (val_u[:, np.newaxis] * val_phi[np.newaxis, :])
            
    w_u = w_gl
    w_phi = np.full(n_phi, dphi)
    
    # Precompute weighted basis for faster computation
    # W[i,j,c] = w_u[i] * w_phi[j] * Y_basis[i,j,c]
    # This lets us compute sum f^p * weights efficiently
    
    # For the gradient, we need:
    # d/dc_j (||f||_4) = (1/4) * ||f||_4^(-3) * 4 * sum W * f^3 * Y_j = ||f||_4^(-3) * sum W * f^3 * Y_j
    # d/dc_j (||f||_2) = (1/2) * ||f||_2^(-1) * 2 * sum W * f * Y_j = ||f||_2^(-1) * sum W * f * Y_j
    # R = ||f||_4 / ||f||_2
    # dR/dc_j = (d||f||_4/dc_j * ||f||_2 - ||f||_4 * d||f||_2/dc_j) / ||f||_2^2
    #         = (sum W * f^3 * Y_j) / ||f||_4^3 / ||f||_2 - ||f||_4/||f||_2^2 * (sum W * f * Y_j) / ||f||_2
    #         = (1/(||f||_4^3 * ||f||_2)) * sum W * f^3 * Y_j - (||f||_4 / ||f||_2^3) * sum W * f * Y_j
    
    # Precompute W * Y_basis for each coefficient
    # WY[j] = sum over (i,k) of w_u[i] * w_phi[k] * Y_basis[i,k,j] ... no, we need per-point
    # Actually, we need: for each point (i,k), the value Y_basis[i,k,j] * w_u[i] * w_phi[k]
    # Let's precompute: weighted_basis[i, k, j] = w_u[i] * w_phi[k] * Y_basis[i, k, j]
    # Shape: (70, 130, 21) = 191100 floats. Fine.
    
    weight_2d = w_u[:, np.newaxis] * w_phi[np.newaxis, :]  # (70, 130)
    weighted_basis = Y_basis * weight_2d[:, :, np.newaxis]  # (70, 130, 21)
    
    # Flatten for faster matrix operations
    # f at all points: (70, 130)
    # f^2: (70, 130)
    # f^3: (70, 130)
    # f^4: (70, 130)
    
    # For gradient computation:
    # sum_{i,k} W[i,k] * f[i,k]^3 * Y_basis[i,k,j]
    # = sum_{i,k} (W[i,k] * f[i,k]^3) * Y_basis[i,k,j]
    # Let g3[i,k] = W[i,k] * f[i,k]^3
    # Let g1[i,k] = W[i,k] * f[i,k]
    # Then d||f||_4/dc_j / (factor) = sum g3[i,k] * Y_basis[i,k,j]
    # And d||f||_2/dc_j / (factor) = sum g1[i,k] * Y_basis[i,k,j]
    # = tensordot with Y_basis
    
    # Precompute Y_basis_flat for tensordot
    # Actually, let's just do it directly
    
    # --- Objective Function with Gradient ---
    
    def compute_ratio_and_grad(c):
        # c shape (21,)
        # f shape (70, 130)
        f = np.tensordot(Y_basis, c, axes=([2], [0]))
        
        f2 = f * f
        f4 = f2 * f2
        
        # Norms
        norm2_sq = np.sum(f2 * weight_2d)
        norm4_4 = np.sum(f4 * weight_2d)
        
        if norm2_sq < 1e-30 or norm4_4 < 1e-30:
            return np.inf, np.zeros(n_coeffs)
            
        norm2 = math.sqrt(norm2_sq)
        norm4 = norm4_4 ** 0.25
        
        R = norm4 / norm2
        
        # Gradient
        # dR/dc_j = (1/(norm4^3 * norm2)) * sum W*f^3*Y_j - (norm4/norm2^3) * sum W*f*Y_j
        
        # g3[i,k] = weight_2d[i,k] * f[i,k]^3
        f3 = f2 * f
        g3 = weight_2d * f3
        g1 = weight_2d * f
        
        # sum over (i,k) of g3[i,k] * Y_basis[i,k,j] for each j
        # = tensordot(g3, Y_basis, axes=([0,1], [0,1]))
        grad_n4 = np.tensordot(g3, Y_basis, axes=([0,1], [0,1]))  # shape (21,)
        grad_n2 = np.tensordot(g1, Y_basis, axes=([0,1], [0,1]))  # shape (21,)
        
        factor1 = 1.0 / (norm4**3 * norm2)
        factor2 = norm4 / (norm2**3)
        
        grad = factor1 * grad_n4 - factor2 * grad_n2
        
        return R, grad
    
    def objective_only(c):
        val, _ = compute_ratio_and_grad(c)
        return val
    
    def objective_jac(c):
        val, grad = compute_ratio_and_grad(c)
        return val, grad
    
    # --- Optimization ---
    
    start_time = time.time()
    deadline = 625.0  # Leave 5s margin for return
    
    # Initial best: uniform coefficients
    best_c = np.ones(n_coeffs)
    best_c /= np.linalg.norm(best_c)
    best_score, _ = compute_ratio_and_grad(best_c)
    
    # Also try some structured initial points
    # m=10 only (most oscillatory)
    c_m10 = np.zeros(n_coeffs)
    c_m10[k] = 1.0  # m=10 -> index 20
    best_score_m10, _ = compute_ratio_and_grad(c_m10)
    if best_score_m10 < best_score:
        best_score = best_score_m10
        best_c = c_m10.copy()
    
    # m=0 only
    c_m0 = np.zeros(n_coeffs)
    c_m0[k] = 1.0  # m=0 -> index 10
    best_score_m0, _ = compute_ratio_and_grad(c_m0)
    if best_score_m0 < best_score:
        best_score = best_score_m0
        best_c = c_m0.copy()
    
    # Random restarts with L-BFGS-B
    n_restarts = 0
    while time.time() - start_time < deadline - 10.0:
        # Random start on the unit sphere
        c0 = np.random.randn(n_coeffs)
        c0 /= np.linalg.norm(c0)
        
        try:
            res = minimize(
                objective_only, 
                c0, 
                jac=lambda c: compute_ratio_and_grad(c)[1],
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            score, _ = compute_ratio_and_grad(res.x)
            if score < best_score:
                best_score = score
                best_c = res.x.copy()
        except Exception:
            pass
        
        n_restarts += 1
        
    # Final Polish on best_c
    try:
        res_final = minimize(
            objective_only, 
            best_c, 
            jac=lambda c: compute_ratio_and_grad(c)[1],
            method='L-BFGS-B', 
            options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-14}
        )
        score, _ = compute_ratio_and_grad(res_final.x)
        if score < best_score:
            best_score = score
            best_c = res_final.x.copy()
    except Exception:
        pass
    
    # Additional polish with more iterations if time permits
    if time.time() - start_time < deadline - 5.0:
        try:
            res_final2 = minimize(
                objective_only, 
                best_c, 
                jac=lambda c: compute_ratio_and_grad(c)[1],
                method='L-BFGS-B', 
                options={'maxiter': 10000, 'gtol': 1e-14, 'ftol': 1e-16}
            )
            score, _ = compute_ratio_and_grad(res_final2.x)
            if score < best_score:
                best_score = score
                best_c = res_final2.x.copy()
        except Exception:
            pass
    
    # Validation
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        return np.ones(n_coeffs)
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs == 0:
            return np.ones(n_coeffs)
        best_c = best_c / max_abs * 1.0
        
    return best_c
