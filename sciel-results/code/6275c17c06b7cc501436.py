import numpy as np
import time
import random
from scipy.optimize import minimize

N = 10
X_GRID, Y_GRID = np.meshgrid(np.arange(N), np.arange(N), indexing='xy')
X_FLAT = X_GRID.flatten()
Y_FLAT = Y_GRID.flatten()

# Pre-compute projection values
U_VALS = X_FLAT - Y_FLAT       # x - y
V_VALS = X_FLAT                # x
W_VALS = X_FLAT + Y_FLAT       # x + y
Z_VALS = X_FLAT + 2 * Y_FLAT   # x + 2y

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

def _objective(params, rank):
    """
    Objective function for the optimizer.
    Minimizes negative ratio.
    params: flattened matrix of size (N * rank)
    rank: rank of the factorization
    """
    U = params[:N*rank].reshape(N, rank)
    V = params[N*rank:].reshape(N, rank)
    
    # Compute log-odds
    log_odds = U @ V.T
    
    # Softmax to get probabilities
    # Subtract max for numerical stability
    log_odds -= np.max(log_odds)
    P = np.exp(log_odds)
    P /= np.sum(P)
    
    ratio = _compute_ratio(P.flatten())
    
    # Return negative ratio to minimize
    return -ratio

def _objective_grad(params, rank):
    """
    Gradient of the objective function using numerical differentiation.
    This is slower but robust for low dimensions (20-40 params).
    """
    eps = 1e-5
    grad = np.zeros_like(params)
    base_val = _objective(params, rank)
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        val_plus = _objective(params_plus, rank)
        grad[i] = (val_plus - base_val) / eps
        
    return grad

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin for 600s limit
    
    best_P = np.ones(100) / 100.0
    best_ratio = _compute_ratio(best_P)
    
    rng = np.random.default_rng(42)
    random.seed(42)
    
    # We will try ranks 1, 2, 3
    # Rank 1: 20 params
    # Rank 2: 40 params
    # Rank 3: 60 params
    
    ranks_to_try = [1, 2, 3]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 30:
            break
            
        n_params = 2 * N * rank
        
        # Multiple restarts for each rank
        n_restarts = 10 if rank == 1 else 5
        
        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize parameters
            # Scale initialization to avoid extreme values
            scale = 1.0
            params_init = rng.normal(0, scale, n_params)
            
            # Run L-BFGS-B
            try:
                res = minimize(
                    _objective, 
                    params_init, 
                    args=(rank,), 
                    method='L-BFGS-B', 
                    jac=_objective_grad, # Using numerical gradient for robustness
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                # Extract best P
                params_best = res.x
                U = params_best[:N*rank].reshape(N, rank)
                V = params_best[N*rank:].reshape(N, rank)
                log_odds = U @ V.T
                log_odds -= np.max(log_odds)
                P_candidate = np.exp(log_odds)
                P_candidate /= np.sum(P_candidate)
                
                ratio = _compute_ratio(P_candidate.flatten())
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_P = P_candidate.flatten()
                    
            except Exception as e:
                pass
                
    # Final refinement: If we have time, run a quick coordinate descent on the best found
    # This helps clean up any numerical artifacts from the factorization
    if time.time() < deadline - 5:
        P_final = best_P.copy()
        # Simple local search to polish
        # We'll just do a few random perturbations and keep the best
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            # Perturb
            noise = rng.normal(0, 0.01, 100)
            P_new = P_final + noise
            P_new = np.clip(P_new, 0, None)
            P_new /= np.sum(P_new)
            
            ratio_new = _compute_ratio(P_new)
            if ratio_new > _compute_ratio(P_final):
                P_final = P_new
                
        final_ratio = _compute_ratio(P_final)
        if final_ratio > best_ratio:
            best_P = P_final
            best_ratio = final_ratio

    # Convert best_P to 10x10 array
    P_out = best_P.reshape((10, 10))
    P_out = np.clip(P_out, 0, None)
    total = np.sum(P_out)
    if total <= 1e-9:
        P_out = np.ones((10, 10)) / 100.0
    else:
        P_out = P_out / total
        
    return P_out

# EVOLVE-BLOCK-END
