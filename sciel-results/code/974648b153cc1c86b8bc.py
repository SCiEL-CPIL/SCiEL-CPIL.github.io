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
    
    # Compute S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    return np.all(np.abs(S - 1.0) < tol)

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

def _block_decomposition_1x5x5(n, m, p):
    """
    Construct a block-diagonal decomposition by treating the n=3 dimension
    as independent 1x5x5 multiplications.
    
    For 1x5x5, we use a structured decomposition. Since we don't have a
    guaranteed rank-19 decomposition for 1x5x5, we use a smarter structured
    approach: we exploit that for 1xm x mxp, the tensor has special structure.
    
    Actually, let's use a known technique: for 1x5x5, we can potentially use
    rank around 25 (schoolbook) or less. Let's try to find a good decomposition
    for 1x5x5 first, then concatenate.
    
    For now, let's construct a valid decomposition using a structured approach
    that might have lower rank than full schoolbook.
    
    For 1x5x5: U is (5, r), V is (25, r), W is (5, r)
    The tensor is T[j,k] = sum_c U[j,c] * V[j*p+k,c] * W[k,c]
    
    Let's try a rank-25 decomposition first (schoolbook for 1x5x5), then try
    to optimize it down.
    """
    # For each of the n=3 blocks, we have a 1x5x5 multiplication
    # Total rank = 3 * rank_1x5x5
    
    # Start with schoolbook for each 1x5x5 block
    r1 = m * p  # 25 for 1x5x5 schoolbook
    total_rank = n * r1  # 75, same as full schoolbook
    
    # This is the same as schoolbook, so let's try a different approach:
    # Use a structured low-rank approximation for 1x5x5
    
    # For 1x5x5, let's try rank 20 (below schoolbook 25)
    # We'll optimize this
    
    return None  # Will be handled by optimization

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

def _extract_result(x, n, m, p, rank):
    """Extract and verify the result from optimization."""
    U_flat = x[:n*m*rank]
    V_flat = x[n*m*rank : n*m*rank + m*p*rank]
    W_flat = x[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
    
    U = U_flat.reshape(n*m, rank)
    V = V_flat.reshape(m*p, rank)
    W = W_flat.reshape(p*n, rank)
    
    U_list = [[float(U[i,c]) for c in range(rank)] for i in range(n*m)]
    V_list = [[float(V[i,c]) for c in range(rank)] for i in range(m*p)]
    W_list = [[float(W[i,c]) for c in range(rank)] for i in range(p*n)]
    
    if _verify_fast(U_list, V_list, W_list, n, m, p, tol=1e-8):
        return U_list, V_list, W_list
    else:
        return None, None, None

def _optimize_decomposition(n, m, p, rank, seed=42, max_iter=1000, tol=1e-10, time_budget=20.0):
    """Use gradient-based optimization to find a rank-rank decomposition."""
    
    def objective(x):
        err, _ = _vectorized_objective(x, n, m, p, rank)
        return err
    
    def gradient(x):
        return _vectorized_gradient(x, n, m, p, rank)
    
    np.random.seed(seed)
    scale = 1.0 / np.sqrt(rank)
    
    # Structured initialization: start near schoolbook-like structure
    U_init = np.zeros((n, m, rank))
    V_init = np.zeros((m, p, rank))
    W_init = np.zeros((p, n, rank))
    
    # Initialize with small random perturbations around a structured pattern
    for c in range(rank):
        U_init[:, :, c] = np.random.randn(n, m) * scale
        V_init[:, :, c] = np.random.randn(m, p) * scale
        W_init[:, :, c] = np.random.randn(p, n) * scale
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    start_time = time.time()
    
    # Try L-BFGS-B
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
    
    elapsed = time.time() - start_time
    
    if result.fun < tol:
        U_r, V_r, W_r = _extract_result(result.x, n, m, p, rank)
        if U_r is not None:
            return U_r, V_r, W_r
    
    return None, None, None

def _try_rank_with_restarts(n, m, p, target_rank, num_restarts=5, time_per_attempt=15.0, start_seed=42):
    """Try to find a decomposition of given rank with multiple restarts."""
    for restart in range(num_restarts):
        if time.time() > time.time() + time_per_attempt:  # This is a placeholder
            break
        seed = start_seed + restart * 137
        U_r, V_r, W_r = _optimize_decomposition(
            n, m, p, target_rank, seed=seed, max_iter=800, tol=1e-10, 
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
    best_result = {"U": U, "V": W, "W": W}
    best_result["V"] = V
    
    # Theoretical lower bound is max(n*m, m*p, p*n) = 25
    # Try ranks from lower bound upward
    
    # Strategy: try ranks in order, prioritizing middle range
    # For 3x5x5, known good ranks are around 35-50
    
    # First pass: quick attempts on all ranks
    for target_rank in range(30, best_rank):
        if time.time() > deadline - 20:
            break
        
        U_r, V_r, W_r, found_rank = _try_rank_with_restarts(
            n, m, p, target_rank, num_restarts=3, time_per_attempt=15.0, start_seed=42
        )
        
        if U_r is not None:
            best_rank = target_rank
            best_result = {"U": U_r, "V": V_r, "W": W_r}
    
    # Second pass: if we found something, try to improve
    if best_rank < n * m * p:
        for target_rank in range(30, best_rank):
            if time.time() > deadline - 10:
                break
            
            U_r, V_r, W_r, found_rank = _try_rank_with_restarts(
                n, m, p, target_rank, num_restarts=4, time_per_attempt=20.0, start_seed=1000
            )
            
            if U_r is not None:
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
    
    return best_result
# EVOLVE-BLOCK-END
