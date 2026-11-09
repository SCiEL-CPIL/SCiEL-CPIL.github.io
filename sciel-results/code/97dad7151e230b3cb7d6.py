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
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_t[i, j, :] * V_t[j, k, :] * W_t[k, i, :]
                s = np.sum(prod)
                if abs(s - 1.0) > tol:
                    return False
    return True

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
    
    # Compute all dot products: for each (i,j,k), sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # We can do this efficiently using einsum
    # U[i,j,c] * V[j,k,c] * W[k,i,c] summed over c
    # First compute UV[i,j,k,c] = U[i,j,c] * V[j,k,c]
    # Then sum over c with W[k,i,c]
    
    # More efficient: for each c, compute the contribution
    err = 0.0
    for c in range(rank):
        Uc = U[:, :, c]  # (n, m)
        Vc = V[:, :, c]  # (m, p)
        Wc = W[:, :, c]  # (p, n)
        
        # For each (i,j,k): Uc[i,j] * Vc[j,k] * Wc[k,i]
        # This is like a tensor contraction
        # Temp[i,j,k] = Uc[i,j] * Vc[j,k] * Wc[k,i]
        # err += sum over i,j,k of (Temp - 1/rank_adjusted)^2 ... no, that's not right
        
        # Actually we need sum_c first, then square
        # Let's compute the full sum differently
        pass
    
    # Better approach: compute S[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # Using einsum: 'ijc,jkc,kic->ijk'
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
        # dU[i,j,c] = sum_k dS[i,j,k] * V[j,k,c] * W[k,i,c]
        # = sum_k dS[i,j,k] * (V[:,k,c] * W[k,i,c]) for fixed i,j
        # Let's compute this as: for each (i,j), sum over k of dS[i,j,k] * V[j,k,c] * W[k,i,c]
        
        # Vc[j,k] = V[j,k,c], Wc[k,i] = W[k,i,c]
        Vc = V[:, :, c]  # (m, p)
        Wc = W[:, :, c]  # (p, n)
        
        # For fixed c: dU[i,j,c] = sum_k dS[i,j,k] * Vc[j,k] * Wc[k,i]
        # This is like: dS[i,j,:] * (Vc[j,:] * Wc[:,i])
        # = dS[i,j,:] dot (Vc[j,:] * Wc[:,i])
        
        # More efficiently:
        # dU_c[i,j] = sum_k dS[i,j,k] * Vc[j,k] * Wc[k,i]
        # = (dS * Wc^T) contracted with Vc... 
        # Let's just do it with einsum or loops
        # dU_c[i,j] = einsum('ijk,jk,ki->ij', dS, Vc, Wc)
        dU[:, :, c] = np.einsum('ijk,jk,ki->ij', dS, Vc, Wc)
        
        # dV[j,k,c] = sum_i dS[i,j,k] * U[i,j,c] * W[k,i,c]
        Uc = U[:, :, c]  # (n, m)
        # dV_c[j,k] = einsum('ijk,ij,ki->jk', dS, Uc, Wc)
        dV[:, :, c] = np.einsum('ijk,ij,ki->jk', dS, Uc, Wc)
        
        # dW[k,i,c] = sum_j dS[i,j,k] * U[i,j,c] * V[j,k,c]
        # dW_c[k,i] = einsum('ijk,ij,jk->ki', dS, Uc, Vc)
        dW[:, :, c] = np.einsum('ijk,ij,jk->ki', dS, Uc, Vc)
    
    return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

def _optimize_decomposition(n, m, p, rank, seed=42, max_iter=2000, tol=1e-10, time_budget=25.0):
    """Use gradient-based optimization with vectorized ops to find a rank-rank decomposition."""
    
    def objective(x):
        err, _ = _vectorized_objective(x, n, m, p, rank)
        return err
    
    def gradient(x):
        return _vectorized_gradient(x, n, m, p, rank)
    
    # Random initialization with different scales
    np.random.seed(seed)
    scale = 1.0 / np.sqrt(rank)
    U_init = np.random.randn(n, m, rank) * scale
    V_init = np.random.randn(m, p, rank) * scale
    W_init = np.random.randn(p, n, rank) * scale
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    start_time = time.time()
    
    # Phase 1: Coarse optimization
    result1 = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                      options={'maxiter': max_iter // 2, 'ftol': 1e-8, 'gtol': 1e-5, 'maxls': 30})
    
    if time.time() - start_time > time_budget * 0.6:
        # Check if we're close enough
        if result1.fun < tol:
            return _extract_result(result1.x, n, m, p, rank)
        return None, None, None, result1.fun, time.time() - start_time
    
    # Phase 2: Fine optimization from phase 1 result
    result2 = minimize(objective, result1.x, jac=gradient, method='L-BFGS-B', 
                      options={'maxiter': max_iter // 2, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
    
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
    if _verify_fast(U_list, V_list, W_list, n, m, p, tol=1e-8):
        return U_list, V_list, W_list, 0.0, 0.0
    else:
        return None, None, None, 0.0, 0.0

def _try_rank_with_restarts(n, m, p, target_rank, num_restarts=5, time_per_attempt=20.0, start_seed=42):
    """Try to find a decomposition of given rank with multiple restarts."""
    for restart in range(num_restarts):
        seed = start_seed + restart * 137
        U_r, V_r, W_r, obj_val, elapsed = _optimize_decomposition(
            n, m, p, target_rank, seed=seed, max_iter=1500, tol=1e-10, 
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
    
    # Theoretical lower bound is max(n*m, m*p, p*n) = 25
    # Try ranks from lower bound upward, with more effort on lower ranks
    
    # Strategy: try ranks in order, spending more time on promising ones
    # For 3x5x5, known good ranks are around 35-45
    # Try from 25 to best_rank, but prioritize middle range
    
    # First pass: quick attempts on all ranks
    for target_rank in range(25, best_rank):
        if time.time() > deadline - 20:
            break
        
        # Quick attempt with fewer restarts
        U_r, V_r, W_r, found_rank = _try_rank_with_restarts(
            n, m, p, target_rank, num_restarts=2, time_per_attempt=15.0, start_seed=42
        )
        
        if U_r is not None:
            best_rank = target_rank
            best_result = {"U": U_r, "V": V_r, "W": W_r}
    
    # Second pass: if we found something, try to improve with more restarts
    if best_rank < n * m * p:
        for target_rank in range(25, best_rank):
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
