"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import minimize

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Fast verification using numpy."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    return np.all(np.abs(S - 1.0) <= tol)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W, rank

def _vectorized_objective(x, n, m, p, rank):
    """Vectorized objective: sum over all (i,j,k) of (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2"""
    U_flat = x[:n*m*rank]
    V_flat = x[n*m*rank : n*m*rank + m*p*rank]
    W_flat = x[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
    
    U = U_flat.reshape(n, m, rank)
    V = V_flat.reshape(m, p, rank)
    W = W_flat.reshape(p, n, rank)
    
    S = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    err = np.sum((S - 1.0) ** 2)
    return err, S

def _vectorized_gradient(x, n, m, p, rank):
    """Vectorized gradient computation."""
    U_flat = x[:n*m*rank]
    V_flat = x[n*m*rank : n*m*rank + m*p*rank]
    W_flat = x[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
    
    U = U_flat.reshape(n, m, rank)
    V = V_flat.reshape(m, p, rank)
    W = W_flat.reshape(p, n, rank)
    
    S = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    dS = 2.0 * (S - 1.0)  # (n, m, p)
    
    # dU[i,j,c] = sum_k dS[i,j,k] * V[j,k,c] * W[k,i,c]
    # dV[j,k,c] = sum_i dS[i,j,k] * U[i,j,c] * W[k,i,c]
    # dW[k,i,c] = sum_j dS[i,j,k] * U[i,j,c] * V[j,k,c]
    
    dU = np.zeros((n, m, rank))
    dV = np.zeros((m, p, rank))
    dW = np.zeros((p, n, rank))
    
    for c in range(rank):
        Vc = V[:, :, c]  # (m, p)
        Wc = W[:, :, c]  # (p, n)
        Uc = U[:, :, c]  # (n, m)
        
        dU[:, :, c] = np.einsum('ijk,jk,ki->ij', dS, Vc, Wc)
        dV[:, :, c] = np.einsum('ijk,ij,ki->jk', dS, Uc, Wc)
        dW[:, :, c] = np.einsum('ijk,ij,jk->ki', dS, Uc, Vc)
    
    return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

def _structured_init(n, m, p, rank, seed=42):
    """Create structured initialization based on Kronecker-like patterns."""
    np.random.seed(seed)
    
    U = np.zeros((n, m, rank))
    V = np.zeros((m, p, rank))
    W = np.zeros((p, n, rank))
    
    # Try different structured patterns
    pattern = seed % 3
    
    if pattern == 0:
        # Block diagonal style
        blocks = min(n, m, p)
        for b in range(blocks):
            for c in range(rank // blocks + 1):
                idx = b * (rank // blocks + 1) + c
                if idx < rank:
                    U[b, b, idx] = 1.0
                    V[b, b, idx] = 1.0
                    W[b, b, idx] = 1.0
        # Add random noise
        U += np.random.randn(n, m, rank) * 0.1
        V += np.random.randn(m, p, rank) * 0.1
        W += np.random.randn(p, n, rank) * 0.1
    elif pattern == 1:
        # Cyclic pattern
        for c in range(rank):
            for i in range(n):
                U[i, (i + c) % m, c] = 1.0 / np.sqrt(rank)
            for j in range(m):
                V[j, (j + c) % p, c] = 1.0 / np.sqrt(rank)
            for k in range(p):
                W[k, (k + c) % n, c] = 1.0 / np.sqrt(rank)
        U += np.random.randn(n, m, rank) * 0.05
        V += np.random.randn(m, p, rank) * 0.05
        W += np.random.randn(p, n, rank) * 0.05
    else:
        # Random with scaling
        scale = 1.0 / np.sqrt(rank)
        U = np.random.randn(n, m, rank) * scale
        V = np.random.randn(m, p, rank) * scale
        W = np.random.randn(p, n, rank) * scale
    
    return U, V, W

def _optimize_decomposition(n, m, p, rank, seed=42, max_iter=2000, tol=1e-8, time_budget=25.0):
    """Use gradient-based optimization with vectorized ops to find a rank-rank decomposition."""
    
    def objective(x):
        err, _ = _vectorized_objective(x, n, m, p, rank)
        return err
    
    def gradient(x):
        return _vectorized_gradient(x, n, m, p, rank)
    
    # Structured initialization
    U_init, V_init, W_init = _structured_init(n, m, p, rank, seed)
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    start_time = time.time()
    
    # Phase 1: Coarse optimization with more iterations
    result1 = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-4, 'maxls': 50})
    
    elapsed1 = time.time() - start_time
    
    if result1.fun < tol:
        return _extract_result(result1.x, n, m, p, rank)
    
    if time.time() - start_time > time_budget * 0.7:
        return None, None, None, result1.fun, elapsed1
    
    # Phase 2: Fine optimization from phase 1 result
    result2 = minimize(objective, result1.x, jac=gradient, method='L-BFGS-B', 
                      options={'maxiter': max_iter // 2, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 30})
    
    elapsed = time.time() - start_time
    
    best_result = result2 if result2.fun < result1.fun else result1
    
    if best_result.fun < tol:
        return _extract_result(best_result.x, n, m, p, rank)
    else:
        return None, None, None, best_result.fun, elapsed

def _extract_result(x, n, m, p, rank):
    """Extract and verify the result from optimization."""
    U_flat = x[:n*m*rank]
    V_flat = x[n*m*rank : n*m*rank + m*p*rank]
    W_flat = x[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
    
    U = U_flat.reshape(n*m, rank)
    V = V_flat.reshape(m*p, rank)
    W = W_flat.reshape(p*n, rank)
    
    # Convert to list of lists
    U_list = [[float(U[i,c]) for c in range(rank)] for i in range(n*m)]
    V_list = [[float(V[i,c]) for c in range(rank)] for i in range(m*p)]
    W_list = [[float(W[i,c]) for c in range(rank)] for i in range(p*n)]
    
    # Verify
    if _verify_fast(U_list, V_list, W_list, n, m, p, tol=1e-7):
        return U_list, V_list, W_list, 0.0, 0.0
    else:
        return None, None, None, 0.0, 0.0

def _try_rank_with_restarts(n, m, p, target_rank, num_restarts=5, time_per_attempt=20.0, start_seed=42):
    """Try to find a decomposition of given rank with multiple restarts."""
    for restart in range(num_restarts):
        seed = start_seed + restart * 137
        U_r, V_r, W_r, obj_val, elapsed = _optimize_decomposition(
            n, m, p, target_rank, seed=seed, max_iter=1500, tol=1e-8, 
            time_budget=time_per_attempt
        )
        
        if U_r is not None:
            return U_r, V_r, W_r, target_rank
    
    return None, None, None, None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s to leave margin for 150s limit
    
    # Start with schoolbook
    U, V, W, best_rank = _schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    
    # For 3x5x5, try ranks in strategic order
    # Prioritize middle ranks where solutions are more likely to exist
    # Lower bound is max(n*m, m*p, p*n) = 25
    # Try from higher likely-successful ranks first, then go lower
    
    # Strategy: try ranks from 50 down to 25, but spend more time on promising ranges
    # Known results suggest ranks around 35-45 are achievable
    
    # First pass: try a range of ranks with quick attempts
    # Focus on ranks 35-55 first as these are more likely to have solutions
    ranks_to_try = list(range(50, 24, -1))  # Try from 50 down to 25
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        # Quick attempt with fewer restarts for higher ranks (more likely to succeed)
        if target_rank >= 40:
            num_restarts = 3
            time_per_attempt = 12.0
        else:
            num_restarts = 2
            time_per_attempt = 10.0
            
        U_r, V_r, W_r, found_rank = _try_rank_with_restarts(
            n, m, p, target_rank, num_restarts=num_restarts, 
            time_per_attempt=time_per_attempt, start_seed=42
        )
        
        if U_r is not None:
            best_rank = target_rank
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            # Found a solution, now try to go lower
            # Break and do a second pass for lower ranks
            break
    
    # Second pass: if we found something, try to improve with more effort on lower ranks
    if best_rank < n * m * p:
        # Try ranks below the found rank
        for target_rank in range(best_rank - 1, 24, -1):
            if time.time() > deadline - 15:
                break
            
            # More restarts and time for lower ranks (harder)
            U_r, V_r, W_r, found_rank = _try_rank_with_restarts(
                n, m, p, target_rank, num_restarts=4, 
                time_per_attempt=15.0, start_seed=1000 + target_rank
            )
            
            if U_r is not None:
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
    
    # Third pass: final polish - try to improve the best found rank
    if best_rank < n * m * p and time.time() < deadline - 10:
        # Try the best rank again with more restarts and different seeds
        U_r, V_r, W_r, found_rank = _try_rank_with_restarts(
            n, m, p, best_rank, num_restarts=3, 
            time_per_attempt=10.0, start_seed=5000
        )
        
        if U_r is not None:
            best_result = {"U": U_r, "V": V_r, "W": W_r}
    
    return best_result
# EVOLVE-BLOCK-END
