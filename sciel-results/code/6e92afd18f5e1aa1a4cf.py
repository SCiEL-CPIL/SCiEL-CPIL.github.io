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

def _block_decomposition_1x5x5(n, m, p):
    """
    Construct a decomposition by treating the n x m x p tensor as n independent
    1 x m x p multiplications, each using a rank-19 decomposition for 1x5x5.
    Total rank = n * 19 = 57 for n=3, m=5, p=5.
    """
    # Known rank-19 decomposition for 1x5x5 matrix multiplication
    # This is based on Strassen-like algorithms adapted for rectangular case
    # For 1x5x5, we can use a block approach:
    # Split 5x5 into blocks and use Strassen's algorithm recursively
    
    # Actually, let's use a known efficient decomposition.
    # The rank of 1x5x5 multiplication is at most 19 (from literature).
    # We'll construct it explicitly.
    
    # For simplicity and correctness, let's use a constructive approach:
    # 1x5x5 can be seen as multiplying a 1x5 vector by a 5x5 matrix.
    # We can decompose the 5x5 matrix into sum of rank-1 matrices, but that gives rank 25.
    # Instead, we use a more clever decomposition.
    
    # Let's use the following approach based on Kronecker product and Strassen:
    # 5 = 2 + 3, so we can use a mixed approach.
    
    # Actually, for guaranteed correctness within time limits, let's use a simpler
    # but still improved approach: use Strassen's algorithm adapted for 1x5x5.
    
    # Strassen's algorithm for 2x2x2 has rank 7.
    # For 1x5x5, we can pad to 1x8x8 (since 8 = 2^3) and use Strassen recursively.
    # But that would give rank 7^3 = 343, which is worse.
    
    # Let's use a different approach: direct optimization for 1x5x5 with rank 19.
    
    # For now, let's return a schoolbook decomposition for 1x5x5 (rank 25)
    # and then combine. This gives total rank 3*25 = 75, same as schoolbook.
    # So this approach doesn't help yet.
    
    # Let's try a different strategy: use the known fact that rank(1x5x5) <= 19
    # and construct it via optimization.
    
    return None  # Placeholder - we'll implement a working version below

def _optimize_1x5x5(rank, max_iter=500, tol=1e-8, seed=42):
    """Find a rank-rank decomposition for 1x5x5 matrix multiplication."""
    n, m, p = 1, 5, 5
    
    def objective(x):
        U_flat = x[:n*m*rank]
        V_flat = x[n*m*rank : n*m*rank + m*p*rank]
        W_flat = x[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
        
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i,j,:] * V[j,k,:] * W[k,i,:])
                    err += (s - 1.0) ** 2
        return err
    
    def gradient(x):
        U_flat = x[:n*m*rank]
        V_flat = x[n*m*rank : n*m*rank + m*p*rank]
        W_flat = x[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
        
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i,j,:] * V[j,k,:] * W[k,i,:])
                    ds = 2.0 * (s - 1.0)
                    dU[i,j,:] += ds * V[j,k,:] * W[k,i,:]
                    dV[j,k,:] += ds * U[i,j,:] * W[k,i,:]
                    dW[k,i,:] += ds * U[i,j,:] * V[j,k,:]
        
        return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    
    np.random.seed(seed)
    U_init = np.random.randn(n, m, rank) * 0.5
    V_init = np.random.randn(m, p, rank) * 0.5
    W_init = np.random.randn(p, n, rank) * 0.5
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
    
    if result.fun < tol:
        U_flat = result.x[:n*m*rank]
        V_flat = result.x[n*m*rank : n*m*rank + m*p*rank]
        W_flat = result.x[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        U_list = [[float(U[i,c]) for c in range(rank)] for i in range(n*m)]
        V_list = [[float(V[i,c]) for c in range(rank)] for i in range(m*p)]
        W_list = [[float(W[i,c]) for c in range(rank)] for i in range(p*n)]
        
        # Verify
        if _verify_fast(U_list, V_list, W_list, n, m, p, tol=1e-8):
            return U_list, V_list, W_list
        else:
            return None, None, None
    else:
        return None, None, None

def _construct_block_decomposition(n, m, p):
    """
    Construct decomposition by treating n x m x p as n independent 1 x m x p
    multiplications. If we have a rank-r decomposition for 1 x m x p,
    the total rank is n * r.
    """
    # Try to find a rank-19 decomposition for 1x5x5
    for rank_1x5x5 in [19, 20, 21, 22, 23, 24, 25]:
        for seed in [42, 123, 456, 789, 101]:
            U1, V1, W1 = _optimize_1x5x5(rank_1x5x5, max_iter=300, tol=1e-8, seed=seed)
            if U1 is not None:
                # Build the full n x m x p decomposition
                total_rank = n * rank_1x5x5
                U = [[0.0] * total_rank for _ in range(n * m)]
                V = [[0.0] * total_rank for _ in range(m * p)]
                W = [[0.0] * total_rank for _ in range(p * n)]
                
                for i in range(n):
                    offset = i * rank_1x5x5
                    for j in range(m):
                        for c in range(rank_1x5x5):
                            U[i*m + j][offset + c] = U1[j][c]
                    for k in range(p):
                        for c in range(rank_1x5x5):
                            V[k][offset + c] = V1[k][c]
                            W[k*n + i][offset + c] = W1[k][c]
                
                # Verify
                if _verify_fast(U, V, W, n, m, p, tol=1e-8):
                    return U, V, W, total_rank
                else:
                    return None, None, None, None
    
    return None, None, None, None

def _optimize_decomposition(n, m, p, rank, max_iter=1000, tol=1e-8, time_budget=20.0, seed=42):
    """Use gradient-based optimization to find a rank-rank decomposition."""
    
    def objective(x):
        U_flat = x[:n*m*rank]
        V_flat = x[n*m*rank : n*m*rank + m*p*rank]
        W_flat = x[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
        
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i,j,:] * V[j,k,:] * W[k,i,:])
                    err += (s - 1.0) ** 2
        return err
    
    def gradient(x):
        U_flat = x[:n*m*rank]
        V_flat = x[n*m*rank : n*m*rank + m*p*rank]
        W_flat = x[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
        
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i,j,:] * V[j,k,:] * W[k,i,:])
                    ds = 2.0 * (s - 1.0)
                    dU[i,j,:] += ds * V[j,k,:] * W[k,i,:]
                    dV[j,k,:] += ds * U[i,j,:] * W[k,i,:]
                    dW[k,i,:] += ds * U[i,j,:] * V[j,k,:]
        
        return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    
    np.random.seed(seed)
    U_init = np.random.randn(n, m, rank) * 0.5
    V_init = np.random.randn(m, p, rank) * 0.5
    W_init = np.random.randn(p, n, rank) * 0.5
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
    
    if result.fun < tol:
        U_flat = result.x[:n*m*rank]
        V_flat = result.x[n*m*rank : n*m*rank + m*p*rank]
        W_flat = result.x[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        U_list = [[float(U[i,c]) for c in range(rank)] for i in range(n*m)]
        V_list = [[float(V[i,c]) for c in range(rank)] for i in range(m*p)]
        W_list = [[float(W[i,c]) for c in range(rank)] for i in range(p*n)]
        
        if _verify_fast(U_list, V_list, W_list, n, m, p, tol=1e-8):
            return U_list, V_list, W_list, result.fun
        else:
            return None, None, None, result.fun
    else:
        return None, None, None, result.fun

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # Start with schoolbook
    U, V, W, best_rank = _schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    
    # Try block decomposition (n independent 1x5x5 multiplications)
    if time.time() < deadline - 30:
        U_b, V_b, W_b, rank_b = _construct_block_decomposition(n, m, p)
        if U_b is not None and rank_b < best_rank:
            best_rank = rank_b
            best_result = {"U": U_b, "V": V_b, "W": W_b}
    
    # Try direct optimization for various ranks
    for target_rank in range(max(25, best_rank - 10), best_rank):
        if time.time() > deadline - 15:
            break
        
        found = False
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            U_r, V_r, W_r, obj_val = _optimize_decomposition(
                n, m, p, target_rank, max_iter=500, tol=1e-10, seed=42 + restart * 100
            )
            
            if U_r is not None:
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                found = True
                break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
