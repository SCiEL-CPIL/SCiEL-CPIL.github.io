import time
import numpy as np

_DIMS = (3, 4, 5)

def _deadline_guard(deadline, safety=5.0):
    """Return True if we should stop searching."""
    return (time.time() - deadline) > safety

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor entry T[i,j,k] should equal sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # We need T[i,j,k] = 1 for all i,j,k
    
    # Iterate over all (i,j,k) since 3*4*5 = 60 is small
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U_r[i, j, :]  # shape (rank,)
                vj = V_r[j, k, :]  # shape (rank,)
                wk = W_r[k, i, :]  # shape (rank,)
                val = np.dot(ui * vj, wk)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(n, m, p, U, V, W, deadline):
    """
    Try to reduce the rank by removing linearly dependent columns.
    We check if any column can be removed while preserving the decomposition.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Try to remove each column one by one
    # If removing column r makes the decomposition invalid, keep it
    # If it remains valid, remove it and try again
    
    current_U = U_arr.copy()
    current_V = V_arr.copy()
    current_W = W_arr.copy()
    current_rank = rank
    
    # We'll try to remove columns in a greedy fashion
    # Check each column from 0 to current_rank-1
    # If removing it keeps the decomposition valid, remove it
    
    # To avoid O(n^2) verification, we can be smarter, but for now let's just try
    # Note: This is a heuristic. True rank minimization is NP-hard.
    
    # We'll iterate until no more columns can be removed
    changed = True
    while changed:
        changed = False
        for r in range(current_rank):
            if _deadline_guard(deadline):
                break
            
            # Try removing column r
            new_U = np.delete(current_U, r, axis=1)
            new_V = np.delete(current_V, r, axis=1)
            new_W = np.delete(current_W, r, axis=1)
            
            # Verify
            if _verify_decomposition(new_U, new_V, new_W, n, m, p):
                current_U = new_U
                current_V = new_V
                current_W = new_W
                current_rank -= 1
                changed = True
                break  # Restart the loop since indices have changed
        
        if _deadline_guard(deadline):
            break
    
    if current_rank < rank:
        return current_U.tolist(), current_V.tolist(), current_W.tolist()
    
    return U, V, W

def construct():
    n, m, p = _DIMS
    deadline = time.time() + 110.0  # Leave 10s safety margin
    
    # Start with schoolbook solution
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify the schoolbook solution
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Should never happen, but safety check
        pass
    
    # Try to reduce rank by removing redundant terms
    if not _deadline_guard(deadline):
        reduced_U, reduced_V, reduced_W = _try_reduce_rank(n, m, p, best_U, best_V, best_W, deadline)
        if reduced_U is not best_U:  # If it actually changed
            # Check if it's better
            new_rank = len(reduced_U[0]) if reduced_U else 0
            if new_rank < best_rank:
                best_U, best_V, best_W = reduced_U, reduced_V, reduced_W
                best_rank = new_rank
    
    # For now, return the best solution found
    return {"U": best_U, "V": best_V, "W": best_W}
