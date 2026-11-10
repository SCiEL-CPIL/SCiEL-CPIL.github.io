import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 10
X_GRID, Y_GRID = np.meshgrid(np.arange(N), np.arange(N), indexing='xy')
X_FLAT = X_GRID.flatten()
Y_FLAT = Y_GRID.flatten()

# Pre-compute projection values
U_VALS = X_FLAT - Y_FLAT       # x - y
V_VALS = X_FLAT                # x
W_VALS = X_FLAT + Y_GRID.flatten()       # x + y
Z_VALS = X_FLAT + 2 * Y_GRID.flatten()   # x + 2y

# Unique values and inverse indices for bincount
U_UNIQ, U_INV = np.unique(U_VALS, return_inverse=True)
V_UNIQ, V_INV = np.unique(V_VALS, return_inverse=True)
W_UNIQ, W_INV = np.unique(W_VALS, return_inverse=True)
Z_UNIQ, Z_INV = np.unique(Z_VALS, return_inverse=True)

def _entropy_from_probs(probs):
    """Calculate Shannon entropy in bits, ignoring zeros."""
    mask = probs > 1e-12
    if not np.any(mask):
        return 0.0
    p = probs[mask]
    p_sum = np.sum(p)
    if p_sum <= 1e-12:
        return 0.0
    p = p / p_sum
    return -np.sum(p * np.log2(p))

def _compute_ratio(P_flat):
    """
    Compute the ratio D / M for a given flat probability vector.
    P_flat: 1D array of length 100, non-negative, summing to 1.
    """
    P_flat = np.clip(P_flat, 0, None)
    total = np.sum(P_flat)
    if total <= 1e-9:
        return 0.0
    P_flat = P_flat / total
    
    p_u = np.bincount(U_INV, weights=P_flat, minlength=len(U_UNIQ))
    p_v = np.bincount(V_INV, weights=P_flat, minlength=len(V_UNIQ))
    p_w = np.bincount(W_INV, weights=P_flat, minlength=len(W_UNIQ))
    p_z = np.bincount(Z_INV, weights=P_flat, minlength=len(Z_UNIQ))
    
    h_u = _entropy_from_probs(p_u)
    h_v = _entropy_from_probs(p_v)
    h_w = _entropy_from_probs(p_w)
    h_z = _entropy_from_probs(p_z)
    
    D = h_u
    M = max(h_v, h_w, h_z)
    
    if M <= 1e-9:
        return 0.0
        
    return D / M

def _gmm_to_prob(params, K):
    """
    Convert GMM parameters to a 100-dim probability vector.
    params layout:
    - Weights: K values (softmax normalized)
    - Means: K * 2 values (x, y)
    - Log-Variances: K * 2 values (log(sigma_x^2), log(sigma_y^2))
    - Correlations: K values (tanh mapped to [-1, 1])
    """
    # Unpack
    w_raw = params[:K]
    means = params[K:K+2*K].reshape(K, 2)
    log_vars = params[K+2*K:K+4*K].reshape(K, 2)
    corr_raw = params[K+4*K:K+5*K]
    
    # Normalize weights
    w = np.exp(w_raw - np.max(w_raw))
    w /= np.sum(w)
    
    # Compute covariances
    sigmas = np.exp(log_vars)
    rhos = np.tanh(corr_raw)
    
    # Evaluate GMM on grid
    # Grid points are (x, y) for x,y in 0..9
    # Distance squared for each component
    P = np.zeros(100)
    
    # Vectorized computation for speed
    # X_grid_flat: (100, 2)
    grid_points = np.stack([X_FLAT, Y_FLAT], axis=1)
    
    for k in range(K):
        mean_k = means[k]
        var_k = sigmas[k]
        rho_k = rhos[k]
        
        # Covariance matrix
        # [[sx^2, sx*sy*rho], [sx*sy*rho, sy^2]]
        sx2 = var_k[0]
        sy2 = var_k[1]
        sxy = np.sqrt(sx2 * sy2) * rho_k
        
        # Determinant
        det = sx2 * sy2 - sxy**2
        if det < 1e-12:
            det = 1e-12
            # Regularize
            sxy = 0.0
            
        # Inverse
        inv_det = 1.0 / det
        inv_cov = np.array([
            [sy2 * inv_det, -sxy * inv_det],
            [-sxy * inv_det, sx2 * inv_det]
        ])
        
        # Mahalanobis distance
        diff = grid_points - mean_k
        # (100, 2) @ (2, 2) -> (100, 2)
        quad = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
        
        # Gaussian term
        # exp(-0.5 * quad) / (2 * pi * sqrt(det))
        norm = 2.0 * np.pi * np.sqrt(det)
        gauss = np.exp(-0.5 * quad) / norm
        
        P += w[k] * gauss
        
    # Normalize to sum 1
    P /= np.sum(P)
    return P

def _objective_gmm(params, K):
    P = _gmm_to_prob(params, K)
    ratio = _compute_ratio(P)
    return -ratio

def _objective_gmm_grad(params, K):
    # Numerical gradient for robustness in low dimensions
    eps = 1e-5
    grad = np.zeros_like(params)
    base_val = _objective_gmm(params, K)
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        val_plus = _objective_gmm(params_plus, K)
        grad[i] = (val_plus - base_val) / eps
        
    return grad

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin for 600s limit
    
    best_P = np.ones(100) / 100.0
    best_ratio = _compute_ratio(best_P)
    
    rng = np.random.default_rng(42)
    
    # Try different numbers of Gaussian components
    # K=1 is just a single Gaussian (centered mass, low entropy everywhere)
    # K=2, 3, 4 allow multi-modal structures
    K_list = [2, 3, 4]
    
    for K in K_list:
        if time.time() > deadline - 100:
            break
            
        n_params = 5 * K
        bounds = []
        
        # Weights: [-5, 5] (logit space)
        bounds.extend([(-5, 5)] * K)
        
        # Means: [0, 9]
        bounds.extend([(0, 9)] * (2 * K))
        
        # Log-Vars: [-2, 2] (sigma from 0.13 to 7.4)
        bounds.extend([(-2, 2)] * (2 * K))
        
        # Correlations: [-3, 3] (tanh maps to -0.99 to 0.99)
        bounds.extend([(-3, 3)] * K)
        
        # Strategy 1: Differential Evolution for global exploration
        # DE is robust for small dimensions
        try:
            # Use a smaller population for speed
            de_res = differential_evolution(
                _objective_gmm,
                bounds,
                args=(K,),
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                polish=False  # We will polish manually
            )
            
            P_candidate = _gmm_to_prob(de_res.x, K)
            ratio = _compute_ratio(P_candidate)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_P = P_candidate.copy()
                
        except Exception as e:
            pass
            
        # Strategy 2: Local refinement with L-BFGS-B starting from DE result and random restarts
        if time.time() > deadline - 50:
            continue
            
        # Start from DE result
        starts = [de_res.x]
        
        # Add some random restarts
        for _ in range(3):
            starts.append(np.array([rng.uniform(b[0], b[1]) for b in bounds]))
            
        for start_params in starts:
            if time.time() > deadline - 20:
                break
                
            try:
                res = minimize(
                    _objective_gmm,
                    start_params,
                    args=(K,),
                    method='L-BFGS-B',
                    jac=_objective_gmm_grad,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                
                P_candidate = _gmm_to_prob(res.x, K)
                ratio = _compute_ratio(P_candidate)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_P = P_candidate.copy()
                    
            except Exception as e:
                pass

    # Final refinement: Coordinate descent on the probability vector itself
    # This allows the final P to deviate from the GMM manifold if it helps
    # (e.g., clipping zeros, sharpening peaks)
    if time.time() < deadline - 10:
        P_final = best_P.copy()
        # Simple local search to polish
        # We'll do a few random perturbations and keep the best
        # Use smaller noise for final polish
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            # Perturb
            noise = rng.normal(0, 0.005, 100)
            P_new = P_final + noise
            P_new = np.clip(P_new, 0, None)
            total = np.sum(P_new)
            if total <= 1e-9:
                continue
            P_new /= total
            
            ratio_new = _compute_ratio(P_new)
            if ratio_new > best_ratio:
                best_ratio = ratio_new
                P_final = P_new
                
        P_out = P_final
    else:
        P_out = best_P

    # Convert best_P to 10x10 array
    P_out = P_out.reshape((10, 10))
    P_out = np.clip(P_out, 0, None)
    total = np.sum(P_out)
    if total <= 1e-9:
        P_out = np.ones((10, 10)) / 100.0
    else:
        P_out = P_out / total
        
    return P_out

# EVOLVE-BLOCK-END
