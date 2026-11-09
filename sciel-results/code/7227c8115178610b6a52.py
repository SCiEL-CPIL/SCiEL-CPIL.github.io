"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _verify_fast(U_arr, V_arr, W_arr, n, m, p, tol=1e-9):
    """Vectorized verification: for all (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    rank = U_arr.shape[1]
    if U_arr.shape != (n*m, rank) or V_arr.shape != (m*p, rank) or W_arr.shape != (p*n, rank):
        return False
    
    idx_i = np.repeat(np.arange(n), m * p)
    idx_j = np.tile(np.repeat(np.arange(m), p), n)
    idx_k = np.tile(np.arange(p), n * m)
    
    u_idx = idx_i * m + idx_j
    v_idx = idx_j * p + idx_k
    w_idx = idx_k * n + idx_i
    
    U_sel = U_arr[u_idx]
    V_sel = V_arr[v_idx]
    W_sel = W_arr[w_idx]
    
    s = np.einsum('ir,ir,ir->i', U_sel, V_sel, W_sel)
    
    return bool(np.all(np.abs(s - 1.0) < tol))

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Schoolbook decomposition
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m + j, col] = 1
                V0[j*p + k, col] = 1
                W0[k*n + i, col] = 1
                col += 1
    
    def to_dict(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best = to_dict(U0, V0, W0)
    
    # Try structured lower-rank decompositions
    # Strategy: The tensor T[i,j,k] = delta for multiplication can sometimes be
    # decomposed with fewer than n*m*p rank-1 tensors by exploiting symmetries.
    # For now, try a greedy elimination: start from schoolbook and try to merge
    # pairs of columns if possible.
    
    rng = np.random.default_rng(123)
    
    # Precompute index arrays for fast verification
    idx_i = np.repeat(np.arange(n), m * p)
    idx_j = np.tile(np.repeat(np.arange(m), p), n)
    idx_k = np.tile(np.arange(p), n * m)
    u_idx = idx_i * m + idx_j
    v_idx = idx_j * p + idx_k
    w_idx = idx_k * n + idx_i
    
    def verify_arrays(U, V, W, tol=1e-9):
        s = np.einsum('ir,ir,ir->i', U[u_idx], V[v_idx], W[w_idx])
        return bool(np.all(np.abs(s - 1.0) < tol))
    
    # Greedy: try to remove columns one at a time by solving for adjustments
    cur_U = U0.copy()
    cur_V = V0.copy()
    cur_W = W0.copy()
    cur_rank = rank0
    
    # Try removing columns in random order
    col_order = list(range(rank0))
    rng.shuffle(col_order)
    
    for col_to_remove in col_order:
        if time.time() > deadline:
            break
        if cur_rank <= 1:
            break
        
        # Try to zero out column col_to_remove and see if we can compensate
        # by adjusting other columns. This is a linear problem.
        # For simplicity, just try zeroing it and checking if still valid (won't work)
        # Better: try to find a linear combination that replaces it.
        
        # Simple approach: try setting the column to zero and check
        old_U_col = cur_U[:, col_to_remove].copy()
        old_V_col = cur_V[:, col_to_remove].copy()
        old_W_col = cur_W[:, col_to_remove].copy()
        
        cur_U[:, col_to_remove] = 0
        cur_V[:, col_to_remove] = 0
        cur_W[:, col_to_remove] = 0
        
        if verify_arrays(cur_U, cur_V, cur_W):
            # Column was redundant! Remove it
            mask = np.ones(cur_rank, dtype=bool)
            mask[col_to_remove] = False
            cur_U = cur_U[:, mask]
            cur_V = cur_V[:, mask]
            cur_W = cur_W[:, mask]
            cur_rank -= 1
        else:
            # Revert
            cur_U[:, col_to_remove] = old_U_col
            cur_V[:, col_to_remove] = old_V_col
            cur_W[:, col_to_remove] = old_W_col
    
    if cur_rank < rank0 and verify_arrays(cur_U, cur_V, cur_W):
        best = to_dict(cur_U, cur_V, cur_W)
    
    return best
# EVOLVE-BLOCK-END
