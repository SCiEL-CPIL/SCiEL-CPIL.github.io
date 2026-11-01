"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def _check_decomposition(U, V, W, tol=1e-9):
    """Check if the decomposition exactly decomposes matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 for all (i,j,k)
    # Build tensor: T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    # We can do this by: for each r, add outer product
    T = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]  # shape (n*m,)
        v_col = V[:, r]  # shape (m*p,)
        w_col = W[:, r]  # shape (p*n,)
        for i in range(n):
            for j in range(m):
                uv = u_col[i * m + j] * v_col[j * p:]
                # W for this i: indices k*n+i for k in range(p)
                w_idx = np.array([k * n + i for k in range(p)])
                T[i, j, :] += uv * w_col[w_idx]
    return np.allclose(T, 1.0, atol=tol)

def _is_half_integer(arr, tol=1e-8):
    """Check if all entries are integer or half-integer."""
    arr = np.array(arr, dtype=float)
    doubled = 2.0 * arr
    nearest = np.round(doubled)
    return np.allclose(doubled, nearest, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook as fallback
    def schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a rank-minimal decomposition
    # Known: for 2x5x5, the rank is at most 50 (schoolbook).
    # There might be a lower rank. Let's try rank 49, 48, ...
    
    best_result = None
    best_rank = n * m * p
    
    # Strategy: random initialization of U, V with small integers/half-integers,
    # then solve for W for each (i,k) pair.
    
    def try_rank(rank, num_trials=200):
        nonlocal best_result, best_rank
        for trial in range(num_trials):
            if time.time() > deadline:
                return
            np.random.seed(12345 + trial)
            random.seed(12345 + trial)
            
            # Initialize U and V with random small values (integers or half-integers)
            # Use values from {-1, -0.5, 0, 0.5, 1}
            vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
            U = np.random.choice(vals, size=(n * m, rank))
            V = np.random.choice(vals, size=(m * p, rank))
            
            # Check that U and V don't have all-zero rows (would make decomposition impossible)
            # Actually, we just need the decomposition to work.
            
            # For each (i, k) pair, solve for W[k*n+i, :]
            W = np.zeros((p * n, rank))
            success = True
            for i in range(n):
                for k in range(p):
                    # Build 5 x rank matrix M where M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    # Solve M @ w = ones(m) for w = W[k*n+i, :]
                    # Minimum norm solution: w = M.T @ inv(M @ M.T) @ ones
                    # But M @ M.T may be singular. Use lstsq or pinv.
                    try:
                        # Use pseudo-inverse
                        M_pinv = np.linalg.pinv(M)
                        w = M_pinv @ np.ones(m)
                    except:
                        success = False
                        break
                    W[k * n + i, :] = w
                    
                    # Check if w is half-integer
                    if not _is_half_integer(w, tol=1e-7):
                        success = False
                        break
                if not success:
                    break
            
            if success:
                # Verify the decomposition
                if _check_decomposition(U, V, W, tol=1e-8):
                    # Round to nearest half-integer
                    U = np.round(2 * U) / 2
                    V = np.round(2 * V) / 2
                    W = np.round(2 * W) / 2
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = rank
                    return True
        
        return False
    
    # Try ranks from 49 down to 40 (reasonable range)
    for rank in range(49, 39, -1):
        if time.time() > deadline:
            break
        if try_rank(rank, num_trials=100):
            break
    
    if best_result is not None:
        return best_result
    
    return schoolbook()

# EVOLVE-BLOCK-END
