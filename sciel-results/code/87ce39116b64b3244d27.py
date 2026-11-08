"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Expected: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > tol:
                    return False
    
    # Check all other entries are zero
    T_check = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_check[a, b, c] = 0
    if np.max(np.abs(T_check)) > tol:
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, max_iter=100, tol=1e-9):
    """Try to reduce rank by finding linear dependencies."""
    rank = U.shape[1]
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    
    for _ in range(max_iter):
        # Try to remove each column
        for r in range(rank):
            # Check if column r can be expressed as combination of others
            # This is complex; use a heuristic instead
            pass
        break  # For now, just return as-is
    
    return U_curr, V_curr, W_curr

def _local_search(n, m, p, target_rank, max_iter=500, tol=1e-9, seed=42):
    """Local search for a decomposition of given rank."""
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(n * m, target_rank)
    V = rng.randn(m * p, target_rank)
    W = rng.randn(p * n, target_rank)
    
    # Compute target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Compute current tensor
        T_curr = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((T_curr - T_target) ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < tol:
            break
        
        # Gradient-based update
        # dL/dU[a,r] = sum_{b,c} (T_curr[a,b,c] - T_target[a,b,c]) * V[b,r] * W[c,r]
        residual = T_curr - T_target
        grad_U = np.einsum('abc,br,cr->ar', residual, V, W)
        grad_V = np.einsum('abc,ar,cr->br', residual, U, W)
        grad_W = np.einsum('abc,ar,br->cr', residual, U, V)
        
        # Adaptive learning rate
        lr = 0.01 / (1 + it * 0.001)
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
    
    # Check if we found an exact solution
    if best_err < tol:
        return best_U, best_V, best_W, True
    
    return best_U, best_V, best_W, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p  # 42
    
    # Try to find better decompositions using local search
    for target_rank in range(30, 42):
        if time.time() > deadline - 15:
            break
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            
            U, V, W, success = _local_search(n, m, p, target_rank, max_iter=200, seed=seed)
            
            if success:
                # Verify
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            # Try to go lower
            continue
    
    return best_result
# EVOLVE-BLOCK-END
