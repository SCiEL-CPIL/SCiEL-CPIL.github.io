import numpy as np
import time
import scipy.optimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 110.0 # Safety margin for 120s limit

def _normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return X / norms

def _riesz_energy_and_grad(v, s=12.0):
    # v is shape (3N,)
    X = v.reshape(N, 3)
    # Normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    P = X / norms
    
    # Compute pairwise distances
    # P_i - P_j
    # dist^2 = 2 - 2 * P_i dot P_j
    dot = P @ P.T
    dist2 = 2.0 - 2.0 * dot
    # Avoid zero distance for i=j
    np.fill_diagonal(dist2, 1.0) # Placeholder, we will mask it
    
    # dist = sqrt(dist2)
    # energy = sum_{i != j} dist^{-s}
    # grad_P = sum_{j != i} -s * dist^{-s-2} * (P_i - P_j)
    
    # Vectorized calculation
    # dist2 is (N, N)
    # We need dist^{-s-2}
    # dist = sqrt(dist2)
    # dist^{-s-2} = (dist2)^{-(s+2)/2}
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Calculate distances for i != j
    # To avoid sqrt(0) issues, we only compute for i != j
    # But vectorized ops on full matrix are faster if we handle diagonal carefully
    
    # Let's compute full matrix and zero out diagonal contributions
    # dist2_safe = np.where(mask, dist2, 1.0)
    # dist_inv_pow = (dist2_safe) ** (-(s + 2) / 2.0)
    # dist_inv_pow[~mask] = 0.0 # Zero out diagonal
    
    # Energy = sum_{i != j} dist^{-s} = sum_{i != j} (dist2)^{-s/2}
    # dist2_safe for energy
    dist2_safe = np.where(mask, dist2, 1.0)
    energy_terms = dist2_safe ** (-s / 2.0)
    energy_terms[~mask] = 0.0
    energy = np.sum(energy_terms)
    
    # Gradient w.r.t P
    # G_P_i = sum_{j != i} -s * (dist2)^{-(s+2)/2} * (P_i - P_j)
    # Note: grad of |P_i - P_j|^{-s} w.r.t P_i is -s |P_i - P_j|^{-s-2} (P_i - P_j)
    
    # Coefficients C_{ij} = -s * (dist2)^{-(s+2)/2}
    coeff = np.where(mask, -s * (dist2_safe) ** (-(s + 2) / 2.0), 0.0)
    
    # G_P = coeff @ (P_i - P_j)? No.
    # G_P_i = sum_j C_{ij} (P_i - P_j)
    # This can be written as:
    # G_P_i = P_i * sum_j C_{ij} - sum_j C_{ij} P_j
    # Let S_i = sum_j C_{ij}
    # Let M = C
    # G_P = P * S[:, None] - (M @ P)
    
    S = np.sum(coeff, axis=1)
    G_P = P * S[:, None] - (coeff @ P)
    
    # Map gradient back to v
    # v_i -> P_i = v_i / ||v_i||
    # dP_i/dv_i = (I / ||v_i||) - (v_i v_i^T / ||v_i||^3)
    # G_v_i = ((I / ||v_i||) - (v_i v_i^T / ||v_i||^3)) G_P_i
    
    # norms is shape (N, 1)
    # G_v = G_P / norms - (G_P @ v.T) * v.T / norms**3 ?
    # Let's do it row by row or vectorized.
    # G_P is (N, 3), v is (N, 3), norms is (N, 1)
    
    # Term 1: G_P / norms
    term1 = G_P / norms
    
    # Term 2: (G_P @ v.T) is (N, N) matrix K where K_ij = G_P_i dot v_j
    # We need to multiply this by v.T (3, N) to get (N, 3)?
    # Formula: (v_i v_i^T) G_P_i = v_i (v_i^T G_P_i)
    # v_i^T G_P_i is the dot product of v_i and G_P_i.
    # dots = np.sum(v * G_P, axis=1) # (N,)
    # term2 = v * dots[:, None] / norms**3
    
    dots = np.sum(v.reshape(N, 3) * G_P, axis=1)
    term2 = v.reshape(N, 3) * dots[:, None] / (norms ** 3)
    
    G_v = term1 - term2
    
    return energy, G_v.flatten()

def _covering_radius(points, grid_size=100):
    # Estimate covering radius using a grid
    # Fibonacci grid for test points
    M = grid_size
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    G = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute angular distances
    # G (M, 3), points (N, 3)
    dots = G @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    return np.max(min_dists)

def construct():
    best_points = None
    best_energy = float('inf')
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Multiple restarts
    n_restarts = 10
    
    for k in range(n_restarts):
        if time.time() > START_TIME + TIME_LIMIT:
            break
            
        if k == 0:
            v0 = init_points.flatten()
        else:
            # Random rotation of init_points or random start
            # Random start
            v0 = np.random.randn(N * 3)
        
        # Optimize
        try:
            res = scipy.optimize.minimize(
                lambda v: _riesz_energy_and_grad(v, s=12.0)[0],
                v0,
                jac=lambda v: _riesz_energy_and_grad(v, s=12.0)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
            )
            
            if res.success or res.fun < best_energy:
                v_opt = res.x
                points = _normalize(v_opt.reshape(N, 3))
                energy = res.fun
                
                if energy < best_energy:
                    best_energy = energy
                    best_points = points
                    
        except Exception as e:
            pass
            
    # If no optimization succeeded, return initial
    if best_points is None:
        return init_points
        
    return best_points
