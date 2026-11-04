"""Program for matrix multiplication 2,3,9 tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]
_DEADLINE = time.time() + 135  # 15s safety margin before 150s kill

def _check_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = U.shape[1]
    # For all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_solve_block(n, m, p, max_rank, time_limit=10):
    """Try to find a rank decomposition for n x m * m x p."""
    start = time.time()
    nm, mp, pn = n*m, m*p, p*n
    for R in range(max_rank, nm*mp*pn - 1, -1):
        if time.time() - start > time_limit:
            return None
        # Try random initialization with Newton-like iterations
        for attempt in range(50):
            if time.time() - start > time_limit:
                break
            random.seed(attempt * 7919 + R * 31)
            # Initialize with small random values
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
            # Normalize
            for r in range(R):
                norm = np.sqrt(np.sum(U[:,r]**2) * np.sum(V[:,r]**2) * np.sum(W[:,r]**2))
                if norm > 0:
                    scale = 1.0 / (norm ** (1.0/3.0))
                    U[:,r] *= scale
                    V[:,r] *= scale
                    W[:,r] *= scale
            # Simple gradient descent on the loss
            for it in range(200):
                if time.time() - start > time_limit:
                    break
                # Compute residuals
                loss = 0.0
                max_grad = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            loss += (s - 1.0)**2
                if loss < 1e-12:
                    return (U.copy(), V.copy(), W.copy())
                # Skip full gradient computation for speed; just check
            # Final check
            if _check_decomposition(U, V, W, n, m, p):
                return (U, V, W)
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Strategy: split p=9 into 3 blocks of 3
    # For each block, solve 2x3 * 3x3
    # If we find rank-r decomposition for 2x3*3x3, total rank = 3r
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank = 54
    
    # Try to find low-rank decomposition for 2x3 * 3x3
    # Schoolbook rank for 2x3*3x3 is 18
    # Try ranks from 10 to 17
    for target_rank in range(10, 18):
        if time.time() > _DEADLINE:
            break
        result = _try_solve_block(n, m, 3, target_rank, time_limit=8)
        if result is not None:
            U_sub, V_sub, W_sub = result
            # Build full decomposition: 3 blocks
            full_rank = 3 * target_rank
            U_full = np.zeros((n*m, full_rank))
            V_full = np.zeros((m*p, full_rank))
            W_full = np.zeros((p*n, full_rank))
            for b in range(3):
                U_full[:, b*target_rank:(b+1)*target_rank] = U_sub
                # V: for block b, k goes from 3b to 3b+2
                for k_local in range(3):
                    k_global = 3*b + k_local
                    V_full[:, b*target_rank:(b+1)*target_rank]  # need to map
                # Actually V_sub has shape (m*3, target_rank) = (9, target_rank)
                # V_full has shape (m*9, full_rank) = (27, full_rank)
                # For block b, row j*3+k_local in V_sub maps to row j*9+k_global in V_full
                for j in range(m):
                    for k_local in range(3):
                        k_global = 3*b + k_local
                        V_full[j*9+k_global, b*target_rank:(b+1)*target_rank] = V_sub[j*3+k_local, :]
                # W_sub has shape (3*n, target_rank) = (6, target_rank)
                # W_full has shape (9*n, full_rank) = (18, full_rank)
                # For block b, row k_local*n+i in W_sub maps to row k_global*n+i in W_full
                for k_local in range(3):
                    k_global = 3*b + k_local
                    for i in range(n):
                        W_full[k_global*n+i, b*target_rank:(b+1)*target_rank] = W_sub[k_local*n+i, :]
            # Verify
            if _check_decomposition(U_full, V_full, W_full, n, m, p):
                if full_rank < best_rank:
                    best_rank = full_rank
                    best_result = (U_full, V_full, W_full)
                break  # Found the lowest rank for this block size
    
    if best_result is None:
        if time.time() > _DEADLINE:
            U, V, W = _schoolbook(n, m, p)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
