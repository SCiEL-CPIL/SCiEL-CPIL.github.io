# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from math import lgamma, log, pi

# Constants
K = 10
M_VALUES = np.arange(-K, K + 1)

# Precompute normalization constants N_m for real orthonormal spherical harmonics
N = np.zeros(2 * K + 1)
for i, m in enumerate(M_VALUES):
    abs_m = abs(m)
    log_n_sq = log(21.0 / (4.0 * pi)) + lgamma(11 - abs_m) - lgamma(11 + abs_m)
    N[i] = np.sqrt(np.exp(log_n_sq))

# Quadrature setup
n_theta = 70
n_phi = 130
x_gl, w_gl = np.polynomial.legendre.leggauss(n_theta)
phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

x_grid, phi_grid_2d = np.meshgrid(x_gl, phi_grid, indexing='ij')

# Precompute the basis functions Y_m(x, phi) for all m
# Shape: (21, 70, 130)
Y_basis = np.zeros((21, n_theta, n_phi))

for i, m in enumerate(M_VALUES):
    abs_m = abs(m)
    p_vals = lpmv(abs_m, 10, x_gl)  # shape (70,)
    
    if m == 0:
        Y_basis[i] = N[i] * p_vals[:, np.newaxis]
    elif m > 0:
        cos_term = np.cos(m * phi_grid_2d)
        Y_basis[i] = np.sqrt(2.0) * N[i] * p_vals[:, np.newaxis] * cos_term
    else:
        sin_term = np.sin(abs_m * phi_grid_2d)
        Y_basis[i] = np.sqrt(2.0) * N[i] * p_vals[:, np.newaxis] * sin_term

# Precompute weights for integration
W_flat = (w_gl[:, np.newaxis] * (2.0 * np.pi / n_phi)).flatten()

# Reshape Y_basis to (21, 70*130)
Y_flat = Y_basis.reshape(21, -1)

# Precompute Gram matrix G where G[i,j] = sum_k W_k Y_i(k) Y_j(k)
# This is the L2 inner product matrix. For orthonormal basis, G should be ~I.
# But due to quadrature approximation, it might not be exactly I.
# Let's compute it to be safe, though theoretically G = I for exact quadrature.
# Actually, the basis is orthonormal by construction, so G = I.
# But let's verify or just use the fact that ||f||_2^2 = c^T G c = ||c||^2 if G=I.
# Let's compute G to be sure.
G = (Y_flat * W_flat[np.newaxis, :]).dot(Y_flat.T)  # (21, 21)

# For the L4 norm, we need ||f||_4^4 = sum_k W_k (sum_i c_i Y_i(k))^4
# This is a 4th order tensor contraction. We can write it as:
# ||f||_4^4 = c^T T c where T is not a matrix but a 4th order tensor.
# However, we can compute the gradient analytically.
# Let f_k = sum_i c_i Y_i(k)
# ||f||_4^4 = sum_k W_k f_k^4
# d(||f||_4^4)/dc_j = sum_k W_k * 4 * f_k^3 * Y_j(k) = 4 * sum_k W_k Y_j(k) f_k^3

# We want to minimize R = ||f||_4 / ||f||_2
# Equivalently, minimize R^2 = ||f||_4^2 / ||f||_2^2 = (||f||_4^4)^{1/2} / ||f||_2^2
# Or minimize log(R) = 0.5 * log(||f||_4^4) - log(||f||_2^2)
# Let's minimize R directly or R^2. R is homogeneous of degree 0, so we can fix ||f||_2 = 1.
# If ||f||_2 = 1, then R = ||f||_4. So minimize ||f||_4 subject to ||f||_2 = 1.
# This is a constrained optimization problem.
# Alternatively, minimize R(c) = ||f||_4 / ||f||_2.
# Gradient of R:
# R = N / D where N = ||f||_4, D = ||f||_2
# dR/dc_j = (D * dN/dc_j - N * dD/dc_j) / D^2
# dN/dc_j = (1/4) * (||f||_4^4)^{-3/4} * 4 * sum_k W_k Y_j(k) f_k^3 = N^{-3} * sum_k W_k Y_j(k) f_k^3
# dD/dc_j = (1/2) * (||f||_2^2)^{-1/2} * 2 * sum_k W_k Y_j(k) f_k = D^{-1} * sum_k W_k Y_j(k) f_k
# So dR/dc_j = (D * N^{-3} * S3_j - N * D^{-1} * S1_j) / D^2
# where S3_j = sum_k W_k Y_j(k) f_k^3 and S1_j = sum_k W_k Y_j(k) f_k
# dR/dc_j = (N^{-3} * S3_j / D - N * D^{-2} * S1_j)
# dR/dc_j = (S3_j / (N^3 * D) - N * S1_j / D^2)
# dR/dc_j = (S3_j * D - N^4 * S1_j) / (N^3 * D^2) ... wait, let's re-evaluate.
# R = N/D
# dR = (D dN - N dD)/D^2
# dN/dc_j = (1/4) N^{-3} * 4 S3_j = N^{-3} S3_j
# dD/dc_j = (1/2) D^{-1} * 2 S1_j = D^{-1} S1_j
# dR/dc_j = (D * N^{-3} S3_j - N * D^{-1} S1_j) / D^2
# dR/dc_j = (S3_j / (N^3 D^2) * D - N S1_j / (D^2) * 1/D) ... no
# dR/dc_j = (D * N^{-3} S3_j - N * D^{-1} S1_j) / D^2
# = (S3_j / (N^3 D) - N S1_j / D^2)
# Let's check dimensions. N has dim f, D has dim f. R is dimensionless.
# S3_j has dim f^4 * (1/f) = f^3? No.
# f_k has dim f. W_k has dim 1. Y_j(k) has dim 1.
# S3_j = sum W Y f^3 has dim f^3.
# N = ||f||_4 has dim f. N^3 has dim f^3.
# S3_j / N^3 is dimensionless.
# S1_j = sum W Y f has dim f.
# D = ||f||_2 has dim f.
# S1_j / D is dimensionless.
# So dR/dc_j = (S3_j / (N^3 D) - N S1_j / D^2) is dimensionless. Correct.

def compute_f(c):
    return c @ Y_flat

def compute_grad_R(c):
    f = compute_f(c)
    norm2_sq = np.sum(W_flat * f**2)
    norm4_4 = np.sum(W_flat * f**4)
    
    if norm2_sq <= 0 or norm4_4 <= 0:
        return np.zeros(21)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    # S1_j = sum_k W_k Y_j(k) f_k = (Y_flat.T * W_flat) @ f
    S1 = (Y_flat.T * W_flat) @ f  # shape (21,)
    # S3_j = sum_k W_k Y_j(k) f_k^3 = (Y_flat.T * W_flat) @ (f**3)
    S3 = (Y_flat.T * W_flat) @ (f**3)  # shape (21,)
    
    grad = S3 / (norm4**3 * norm2) - norm4 * S1 / (norm2**2)
    return grad

def evaluate_ratio(c):
    f = compute_f(c)
    norm2_sq = np.sum(W_flat * f**2)
    norm4_4 = np.sum(W_flat * f**4)
    if norm2_sq <= 0 or norm4_4 <= 0:
        return np.inf
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    return norm4 / norm2

def optimize_from_start(c0, time_budget):
    from scipy.optimize import minimize
    
    start_time = time.time()
    
    def objective(c):
        return evaluate_ratio(c)
    
    def gradient(c):
        return compute_grad_R(c)
    
    # Use L-BFGS-B with analytic gradient
    # We can also use trust-constr if needed, but L-BFGS-B is often robust.
    # The problem is unconstrained in c (scale invariance is handled by the ratio).
    # However, c=0 is a singularity. We assume c0 is not zero.
    
    try:
        res = minimize(objective, c0, jac=gradient, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
        return res.x, res.fun
    except Exception:
        return c0, objective(c0)

def refine_local(coeffs, time_left_fn):
    from scipy.optimize import minimize
    
    def objective(c):
        return evaluate_ratio(c)
    
    def gradient(c):
        return compute_grad_R(c)
    
    best_c = coeffs.copy()
    best_val = objective(coeffs)
    
    # L-BFGS-B pass
    try:
        result = minimize(objective, coeffs, jac=gradient, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
        if result.fun < best_val:
            best_c = result.x.copy()
            best_val = result.fun
    except Exception:
        pass
    
    # Nelder-Mead polish (derivative-free, robust to flat regions)
    if time_left_fn() > 2.0:
        try:
            result2 = minimize(objective, best_c, method='Nelder-Mead', 
                              options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            if result2.fun < best_val:
                best_c = result2.x.copy()
                best_val = result2.fun
        except Exception:
            pass
            
    return best_c

def random_search(time_budget):
    best_coeffs = None
    best_val = np.inf
    start_time = time.time()
    
    def time_left():
        return time_budget - (time.time() - start_time)
    
    # Initial candidates
    candidates = []
    
    # 1. All ones (baseline)
    candidates.append(np.ones(21))
    
    # 2. Random Gaussian
    for _ in range(20):
        candidates.append(np.random.randn(21))
        
    # 3. Structured patterns
    candidates.append(np.ones(21) * np.where(np.abs(M_VALUES) <= 2, 1, 0))
    candidates.append(np.ones(21) * np.where(M_VALUES % 2 == 0, 1, 0))
    candidates.append(np.ones(21) * np.where(M_VALUES % 2 != 0, 1, 0))
    candidates.append(np.ones(21) * np.where(np.abs(M_VALUES) >= 8, 1, 0))
    candidates.append(np.ones(21) * np.where(np.abs(M_VALUES) <= 5, 1, 0))
    candidates.append(np.ones(21) * np.where(np.abs(M_VALUES) >= 5, 1, 0))
    
    # 4. Alternating signs patterns
    for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
        c = np.ones(21)
        for i, m in enumerate(M_VALUES):
            c[i] = np.cos(m * phase)
        candidates.append(c)
        
    # 5. Sparse random
    for _ in range(10):
        c = np.zeros(21)
        n_nonzero = np.random.randint(3, 8)
        indices = np.random.choice(21, n_nonzero, replace=False)
        c[indices] = np.random.randn(n_nonzero)
        candidates.append(c)

    # Evaluate initial candidates
    for c in candidates:
        val = evaluate_ratio(c)
        if val < best_val:
            best_val = val
            best_coeffs = c.copy()
            
    # Main search loop
    # We will alternate between:
    # 1. Global random search (restarts)
    # 2. Local refinement from best point
    # 3. Local refinement from random points
    
    while time_left() > 10.0:
        # 1. Refine the current best
        if best_coeffs is not None and time_left() > 15.0:
            refined = refine_local(best_coeffs, time_left)
            refined_val = evaluate_ratio(refined)
            if refined_val < best_val:
                best_val = refined_val
                best_coeffs = refined.copy()
                
        # 2. Try a few new random starts with local refinement
        if time_left() > 20.0:
            for _ in range(3):
                c_new = np.random.randn(21)
                # Maybe scale it
                c_new = c_new / np.linalg.norm(c_new) * np.sqrt(21) # Normalize to typical scale
                
                # Quick local refine
                try:
                    res = minimize(lambda c: evaluate_ratio(c), c_new, 
                                   jac=lambda c: compute_grad_R(c), 
                                   method='L-BFGS-B', 
                                   options={'maxiter': 50, 'ftol': 1e-8})
                    if res.fun < best_val:
                        best_val = res.fun
                        best_coeffs = res.x.copy()
                except Exception:
                    pass
                    
                if time_left() < 10.0:
                    break

    return best_coeffs

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    max_time = 580.0
    
    try:
        best_coeffs = random_search(max_time)
    except Exception as e:
        best_coeffs = np.ones(21)
        
    # Final validation and cleanup
    if best_coeffs is None or not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21)
        
    if len(best_coeffs) != 21:
        best_coeffs = np.ones(21)
        
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3 or max_c > 1e5:
        if max_c > 0:
            best_coeffs = best_coeffs / max_c * 1.0
        else:
            best_coeffs = np.ones(21)
            
    # Final high-precision polish if time permits
    elapsed = time.time() - start_time
    if elapsed < max_time - 30 and best_coeffs is not None and np.all(np.isfinite(best_coeffs)):
        try:
            refined = refine_local(best_coeffs, lambda: max_time - (time.time() - start_time))
            if np.all(np.isfinite(refined)):
                val_orig = evaluate_ratio(best_coeffs)
                val_ref = evaluate_ratio(refined)
                if val_ref < val_orig:
                    best_coeffs = refined
        except:
            pass
    
    # Final validation after polish
    if best_coeffs is None or not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21)
    if len(best_coeffs) != 21:
        best_coeffs = np.ones(21)
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3 or max_c > 1e5:
        if max_c > 0:
            best_coeffs = best_coeffs / max_c * 1.0
        else:
            best_coeffs = np.ones(21)
            
    return best_coeffs

# EVOLVE-BLOCK-END
