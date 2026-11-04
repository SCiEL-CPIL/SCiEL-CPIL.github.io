"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to 3D tensors for easier einsum
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # Compute tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    target = np.ones((n, m, p))
    
    return np.allclose(T, target, atol=tol)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _als_decompose(n, m, p, rank, max_iters=500, seed=42, tol=1e-10, deadline=None):
    """
    Use Alternating Least Squares to find a rank-`rank` decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Constraint: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    prev_loss = float('inf')
    
    for it in range(max_iters):
        if deadline and time.time() > deadline - 2:
            break
            
        # --- Update U ---
        for i in range(n):
            for j in range(m):
                B = np.zeros((p, rank))
                for k in range(p):
                    B[k, :] = V[j*p+k, :] * W[k*n+i, :]
                target = np.ones(p)
                try:
                    U[i*m+j, :], _, _, _ = np.linalg.lstsq(B, target, rcond=None)
                except np.linalg.LinAlgError:
                    pass
        
        # --- Update V ---
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                target = np.ones(n)
                try:
                    V[j*p+k, :], _, _, _ = np.linalg.lstsq(B, target, rcond=None)
                except np.linalg.LinAlgError:
                    pass

        # --- Update W ---
        for k in range(p):
            for i in range(n):
                B = np.zeros((m, rank))
                for j in range(m):
                    B[j, :] = U[i*m+j, :] * V[j*p+k, :]
                target = np.ones(m)
                try:
                    W[k*n+i, :], _, _, _ = np.linalg.lstsq(B, target, rcond=None)
                except np.linalg.LinAlgError:
                    pass

        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        target_tensor = np.ones((n, m, p))
        loss = np.sum((T - target_tensor) ** 2)
        
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
        
    return U, V, W, loss

def _try_optimize_rank(n, m, p, target_rank, max_als_iters=100, seeds=10, deadline=None):
    """Try to find a rank-target_rank decomposition using ALS."""
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for seed in range(seeds):
        if deadline and time.time() > deadline - 5:
            break
            
        U, V, W, loss = _als_decompose(n, m, p, target_rank, max_iters=max_als_iters, seed=seed, deadline=deadline)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if loss < 1e-6:
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            
            if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p, tol=1e-6):
                return U_int.tolist(), V_int.tolist(), W_int.tolist(), loss
            
            U_half = np.round(best_U * 2) / 2
            V_half = np.round(best_V * 2) / 2
            W_half = np.round(best_W * 2) / 2
            
            if _verify(U_half.tolist(), V_half.tolist(), W_half.tolist(), n, m, p, tol=1e-6):
                return U_half.tolist(), V_half.tolist(), W_half.tolist(), loss
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    ranks_to_try = list(range(24, 72))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        time_left = deadline - time.time()
        
        if time_left > 30:
            max_its = 100
            seeds = 10
        elif time_left > 15:
            max_its = 50
            seeds = 5
        else:
            max_its = 20
            seeds = 2
            
        result = _try_optimize_rank(n, m, p, target_rank, max_als_iters=max_its, seeds=seeds, deadline=deadline)
        
        if result is not None:
            U_r, V_r, W_r, loss = result
            if _verify(U_r, V_r, W_r, n, m, p, tol=1e-6):
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                return best_result
    
    return best_result

# EVOLVE-BLOCK-END
