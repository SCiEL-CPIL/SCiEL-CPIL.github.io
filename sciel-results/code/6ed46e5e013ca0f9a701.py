# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv, gamma
import time

def _build_quadrature():
    """Build 70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    dphi = 2.0 * np.pi / n_phi
    return x_gl, w_gl, phi, dphi

def _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=10):
    """
    Build the matrix Y[i,j] where i indexes quadrature points (n_gl * n_phi)
    and j indexes m = -k..k.
    """
    n_gl = len(x_gl)
    n_phi = len(phi)
    n_m = 2 * k + 1
    Y = np.zeros((n_gl * n_phi, n_m), dtype=np.float64)
    
    for m_idx in range(n_m):
        m = m_idx - k
        abs_m = abs(m)
        # Normalization
        N_m = np.sqrt((2 * k + 1) / (4.0 * np.pi) * 
                       float(gamma(k - abs_m + 1)) / float(gamma(k + abs_m + 1)))
        # Associated Legendre with Condon-Shortley phase
        P = lpmv(abs_m, k, x_gl)  # shape (n_gl,)
        
        if m == 0:
            vals = N_m * P  # shape (n_gl,)
            Y[:, m_idx] = np.repeat(vals, n_phi)
        elif m > 0:
            cos_m = np.cos(m * phi)  # shape (n_phi,)
            vals = np.sqrt(2.0) * N_m * P[:, np.newaxis] * cos_m[np.newaxis, :]
            Y[:, m_idx] = vals.ravel()
        else:
            sin_m = np.sin(abs_m * phi)  # shape (n_phi,)
            vals = np.sqrt(2.0) * N_m * P[:, np.newaxis] * sin_m[np.newaxis, :]
            Y[:, m_idx] = vals.ravel()
    
    return Y

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin
    
    k = 10
    x_gl, w_gl, phi, dphi = _build_quadrature()
    Y = _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=k)
    n_points = Y.shape[0]
    
    # Precompute weights
    n_gl = len(w_gl)
    n_phi = Y.shape[0] // n_gl
    weights = np.repeat(w_gl, n_phi) * dphi
    
    # Strategy: Optimize over the subspace of even-m modes (m = 0, ±2, ±4, ±6, ±8, ±10).
    # This subspace has dimension 1 + 2*5 = 11? No.
    # m=0: 1 coeff
    # m=±2: 2 coeffs
    # m=±4: 2 coeffs
    # m=±6: 2 coeffs
    # m=±8: 2 coeffs
    # m=±10: 2 coeffs
    # Total 11 coefficients. This is a reduction from 21 to 11.
    # The directive mentioned "1-dimensional problem" but also listed multiple modes.
    # Let's stick to the even-m subspace as it preserves the symmetry f(phi) = f(phi+pi).
    
    # Indices for even m in the full 21-vector (m = -10..10)
    # m values: -10, -9, ..., 0, ..., 9, 10
    # Even m: -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10
    even_m_indices = [i for i, m in enumerate(range(-k, k+1)) if m % 2 == 0]
    
    # We will optimize these 11 coefficients.
    n_sub = len(even_m_indices)
    
    def objective_and_grad_sub(sub_coeffs):
        # Map sub_coeffs to full coeffs
        full_coeffs = np.zeros(2 * k + 1)
        full_coeffs[even_m_indices] = sub_coeffs
        
        f = Y @ full_coeffs
        f2 = f * f
        f4 = f2 * f2
        norm2_sq = np.dot(weights, f2)
        norm4_sq = np.dot(weights, f4)
        if norm2_sq <= 0:
            return 1e10, np.zeros_like(sub_coeffs)
        norm2 = np.sqrt(norm2_sq)
        norm4 = np.sqrt(norm4_sq)
        R = norm4 / norm2
        
        # Gradient with respect to full coeffs
        grad_full = np.zeros(2 * k + 1)
        grad_term1 = Y.T @ (weights * f**3)
        grad_term2 = Y.T @ (weights * f)
        grad_full = (2.0 / (norm4 * norm2)) * grad_term1 - (2.0 * norm4 / norm2**3) * grad_term2
        
        # Project gradient to subspace
        grad_sub = grad_full[even_m_indices]
        
        return R, grad_sub

    # Initial guess: Zonal harmonic (m=0) is a good start
    x_sub = np.zeros(n_sub)
    x_sub[even_m_indices.index(10)] = 1.0 # m=0 is at index 10 (0-based: -10 is 0, 0 is 10)
    
    best_coeffs_sub = x_sub.copy()
    best_val = objective_and_grad_sub(x_sub)[0]
    
    # Gradient descent with adaptive step size
    lr = 0.01
    n_steps = 50000
    patience = 5000
    no_improve = 0
    
    for step in range(n_steps):
        if time.time() > deadline - 30.0:
            break
        
        val, grad = objective_and_grad_sub(x_sub)
        
        if val < best_val:
            best_val = val
            best_coeffs_sub = x_sub.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > patience:
            lr *= 0.5
            no_improve = 0
            if lr < 1e-10:
                break
        
        x_new = x_sub - lr * grad
        
        if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
            lr *= 0.1
            x_new = x_sub - lr * grad
            if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
                break
        
        x_sub = x_new
    
    # Random restarts in the subspace
    np.random.seed(42)
    for restart in range(10):
        if time.time() > deadline - 60.0:
            break
        
        x_sub = np.random.randn(n_sub)
        # Normalize
        full_coeffs = np.zeros(2 * k + 1)
        full_coeffs[even_m_indices] = x_sub
        f = Y @ full_coeffs
        norm2 = np.sqrt(np.dot(weights, f * f))
        if norm2 > 0:
            x_sub = x_sub / norm2
        
        lr = 0.01
        no_improve = 0
        
        for step in range(20000):
            if time.time() > deadline - 30.0:
                break
            
            val, grad = objective_and_grad_sub(x_sub)
            
            if val < best_val:
                best_val = val
                best_coeffs_sub = x_sub.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > patience:
                lr *= 0.5
                no_improve = 0
                if lr < 1e-10:
                    break
            
            x_new = x_sub - lr * grad
            
            if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
                lr *= 0.1
                x_new = x_sub - lr * grad
                if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
                    break
            
            x_sub = x_new
    
    # Map back to full coefficients
    best_coeffs = np.zeros(2 * k + 1)
    best_coeffs[even_m_indices] = best_coeffs_sub
    
    # Validate
    if len(best_coeffs) != 21:
        return np.ones(21)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(21)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        return np.ones(21)
    
    return best_coeffs

# EVOLVE-BLOCK-END
