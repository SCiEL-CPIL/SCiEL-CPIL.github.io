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

def _sparse_search(rng, deadline):
    """
    Sparse combinatorial approach:
    1. Select a subset of cells (support) to maximize H_{x-y} while minimizing max(H_x, H_{x+y}, H_{x+2y})
    2. Assign uniform probability to the selected cells
    3. Refine with local search
    """
    all_cells = np.arange(100)
    best_ratio = 0.0
    best_support = None
    
    # Try different support sizes
    for k in range(5, 50, 3):
        if time.time() > deadline - 60:
            break
        
        # Multiple random restarts for each size
        n_restarts = max(3, 20 - k // 10)
        
        for r in range(n_restarts):
            if time.time() > deadline - 30:
                break
            
            # Random subset
            support = rng.choice(all_cells, size=k, replace=False)
            
            # Evaluate uniform distribution on this support
            P_flat = np.zeros(100)
            P_flat[support] = 1.0 / k
            ratio = _compute_ratio(P_flat)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_support = support.copy()
    
    # Local search: add/remove cells from support
    if best_support is not None and time.time() < deadline - 30:
        current_support = best_support.copy()
        current_ratio = best_ratio
        
        for _ in range(200):
            if time.time() > deadline - 15:
                break
            
            improved = False
            
            # Try adding a random cell
            if len(current_support) < 60:
                candidates = np.setdiff1d(all_cells, current_support)
                if len(candidates) > 0:
                    add_cell = rng.choice(candidates)
                    new_support = np.union1d(current_support, [add_cell])
                    P_flat = np.zeros(100)
                    P_flat[new_support] = 1.0 / len(new_support)
                    ratio = _compute_ratio(P_flat)
                    if ratio > current_ratio:
                        current_ratio = ratio
                        current_support = new_support
                        improved = True
            
            # Try removing a random cell
            if len(current_support) > 5:
                remove_idx = rng.integers(len(current_support))
                remove_cell = current_support[remove_idx]
                new_support = np.setdiff1d(current_support, [remove_cell])
                P_flat = np.zeros(100)
                P_flat[new_support] = 1.0 / len(new_support)
                ratio = _compute_ratio(P_flat)
                if ratio > current_ratio:
                    current_ratio = ratio
                    current_support = new_support
                    improved = True
            
            if not improved:
                # Try swapping
                if len(current_support) >= 5 and len(current_support) < 60:
                    candidates = np.setdiff1d(all_cells, current_support)
                    if len(candidates) > 0:
                        add_cell = rng.choice(candidates)
                        remove_idx = rng.integers(len(current_support))
                        remove_cell = current_support[remove_idx]
                        new_support = np.setdiff1d(current_support, [remove_cell])
                        new_support = np.union1d(new_support, [add_cell])
                        P_flat = np.zeros(100)
                        P_flat[new_support] = 1.0 / len(new_support)
                        ratio = _compute_ratio(P_flat)
                        if ratio > current_ratio:
                            current_ratio = ratio
                            current_support = new_support
    
    if best_support is not None:
        P_out = np.zeros(100)
        P_out[best_support] = 1.0 / len(best_support)
        return P_out, best_ratio
    
    return None, 0.0

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # Full 600s limit
    
    best_P = np.ones(100) / 100.0
    best_ratio = _compute_ratio(best_P)
    
    rng = np.random.default_rng(42)
    random.seed(42)
    
    # Phase 1: Sparse combinatorial search
    if time.time() < deadline - 300:
        sparse_P, sparse_ratio = _sparse_search(rng, deadline)
        if sparse_P is not None and sparse_ratio > best_ratio:
            best_P = sparse_P
            best_ratio = sparse_ratio
    
    # Phase 2: Parametric optimization from multiple initializations
    ranks_to_try = [1, 2, 3]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 100:
            break
        
        n_params = 2 * N * rank
        n_restarts = 8 if rank == 1 else 4
        
        for r in range(n_restarts):
            if time.time() > deadline - 50:
                break
            
            # Initialize parameters
            scale = 1.0
            params_init = rng.normal(0, scale, n_params)
            
            # Run L-BFGS-B
            try:
                res = minimize(
                    _objective, 
                    params_init, 
                    args=(rank,), 
                    method='L-BFGS-B', 
                    jac=_objective_grad,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
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
    
    # Phase 3: Final refinement with coordinate descent
    if time.time() < deadline - 10:
        P_final = best_P.copy()
        
        # Coordinate descent: optimize one cell at a time
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for idx in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Try different values for this cell
                old_val = P_final[idx]
                for new_val in [0.0, old_val * 0.5, old_val * 2.0, 0.01, 0.05]:
                    P_test = P_final.copy()
                    P_test[idx] = new_val
                    P_test = np.clip(P_test, 0, None)
                    total = np.sum(P_test)
                    if total <= 1e-9:
                        continue
                    P_test /= total
                    
                    ratio_test = _compute_ratio(P_test)
                    if ratio_test > _compute_ratio(P_final) + 1e-10:
                        P_final = P_test
                        improved = True
                        break
            
            if not improved:
                break
        
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
