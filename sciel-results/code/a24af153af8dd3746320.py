"""Matrix multiplication 3x3x4: rank-minimal exact tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS

def _check_exact_vectorized(U, V, W):
    """Check if the decomposition is exact using vectorized operations."""
    U = np.array(U, dtype=float).reshape(n, m, -1)
    V = np.array(V, dtype=float).reshape(m, p, -1)
    W = np.array(W, dtype=float).reshape(p, n, -1)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < 1e-9

def _to_half_integer(U, V, W):
    """Round entries to nearest half-integer and check validity."""
    U_r = np.round(2 * np.array(U, dtype=float)) / 2.0
    V_r = np.round(2 * np.array(V, dtype=float)) / 2.0
    W_r = np.round(2 * np.array(W, dtype=float)) / 2.0
    
    if _check_exact_vectorized(U_r, V_r, W_r):
        return [U_r.reshape(-1, U_r.shape[-1]).tolist(), 
                V_r.reshape(-1, V_r.shape[-1]).tolist(), 
                W_r.reshape(-1, W_r.shape[-1]).tolist()]
    return None

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(rank, time_budget):
    """Try to find a valid rank-decomposition using vectorized gradient descent."""
    deadline = time.time() + time_budget
    
    # Reshape for easier einsum usage
    # U: (n, m, rank), V: (m, p, rank), W: (p, n, rank)
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        np.random.seed(attempt * 7919 + rank)
        U = np.random.randn(n, m, rank) * 0.5
        V = np.random.randn(m, p, rank) * 0.5
        W = np.random.randn(p, n, rank) * 0.5
        
        # Momentum
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        best_err = float('inf')
        
        for it in range(5000):
            if time.time() > deadline:
                break
            
            # Compute T
            T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            err = T - 1.0
            err_norm = np.linalg.norm(err)
            
            if err_norm < 1e-8:
                break
            
            if err_norm < best_err:
                best_err = err_norm
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Gradients
            # dE/dU[i,j,r] = sum_k 2 * err[i,j,k] * V[j,k,r] * W[k,i,r]
            gU = 2 * np.einsum('ijk,jkr,kir->ijr', err, V, W)
            gV = 2 * np.einsum('ijk,ijr,kir->jkr', err, U, W)
            gW = 2 * np.einsum('ijk,ijr,jkr->kir', err, U, V)
            
            # Learning rate decay
            lr = 0.01 / (1 + it * 0.001)
            
            # Momentum update
            vU = 0.9 * vU - lr * gU
            vV = 0.9 * vV - lr * gV
            vW = 0.9 * vW - lr * gW
            
            U += vU
            V += vV
            W += vW
            
            # Optional: Periodic check for half-integer solution
            if it % 500 == 0:
                result = _to_half_integer(U, V, W)
                if result is not None:
                    return result
        
        # Final check
        result = _to_half_integer(best_U, best_V, best_W)
        if result is not None:
            return result
        
        if _check_exact_vectorized(best_U, best_V, best_W):
            return [best_U.reshape(-1, rank).tolist(), 
                    best_V.reshape(-1, rank).tolist(), 
                    best_W.reshape(-1, rank).tolist()]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    # Try ranks from lower to higher. 
    # Theoretical lower bound for 3x3x3 is 23. For 3x3x4, it's likely slightly higher.
    # Schoolbook is 36.
    ranks_to_try = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    
    # Allocate time based on likelihood
    # Lower ranks are harder, so give them more time? Or just split evenly?
    # Let's try a few low ranks first.
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        
        # Allocate remaining time / remaining ranks
        remaining_ranks = len(ranks_to_try) - ranks_to_try.index(r)
        time_budget = (deadline - time.time()) / remaining_ranks
        time_budget = max(2, min(15, time_budget))
        
        result = _try_rank(r, time_budget)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    return _schoolbook()
