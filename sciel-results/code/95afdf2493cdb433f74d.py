"""Matrix multiplication 4x5x6 tensor decomposition - search for rank-minimal."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact using vectorized computation."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    nm, mp, pn = n * m, m * p, p * n
    
    # Vectorized verification: for each (i,j,k), compute sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Build index arrays
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx  # shape (n*m*p,)
    v_idx = j_idx * p + k_idx  # shape (n*m*p,)
    w_idx = k_idx * n + i_idx  # shape (n*m*p,)
    
    # Compute all dot products at once
    # U_arr[u_idx, :] has shape (n*m*p, rank)
    # V_arr[v_idx, :] has shape (n*m*p, rank)
    # W_arr[w_idx, :] has shape (n*m*p, rank)
    U_sel = U_arr[u_idx]  # (nmp, rank)
    V_sel = V_arr[v_idx]  # (nmp, rank)
    W_sel = W_arr[w_idx]  # (nmp, rank)
    
    # Element-wise product then sum along rank axis
    vals = np.sum(U_sel * V_sel * W_sel, axis=1)  # (nmp,)
    
    max_err = np.max(np.abs(vals - 1.0))
    if max_err > 1e-9:
        # Find the worst entry
        worst_idx = np.argmax(np.abs(vals - 1.0))
        i = i_idx[worst_idx]
        j = j_idx[worst_idx]
        k = k_idx[worst_idx]
        return False, (i, j, k, vals[worst_idx])
    return True, None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _try_strassen_embedding(n, m, p):
    """
    Try to embed the 4x5x6 multiplication into a larger structured decomposition.
    Use a block-diagonal approach with small Strassen-like blocks.
    """
    # This is a heuristic attempt; return None if it doesn't work
    pass

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = len(U[0])
    best_result = (U, V, W)
    
    # Convert to numpy for efficient computation
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Precompute index arrays for vectorized operations
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    def get_tensor_contribution(U_arr, V_arr, W_arr, c):
        """Get the rank-1 tensor contribution of column c as a flat array."""
        return U_arr[u_idx, c] * V_arr[v_idx, c] * W_arr[w_idx, c]
    
    def can_merge(U_arr, V_arr, W_arr, c1, c2):
        """Check if columns c1 and c2 can be merged. Returns (alpha, success)."""
        t1 = get_tensor_contribution(U_arr, V_arr, W_arr, c1)
        t2 = get_tensor_contribution(U_arr, V_arr, W_arr, c2)
        
        nonzero1 = np.abs(t1) > 1e-12
        nonzero2 = np.abs(t2) > 1e-12
        
        if not np.any(nonzero1):
            return 0.0, True  # t1 is zero, can remove c2
        if not np.any(nonzero2):
            return 0.0, True  # t2 is zero, already removable
        
        overlap = nonzero1 & nonzero2
        if not np.any(overlap):
            return 0.0, False
        
        # Check if t2 = alpha * t1 where both are nonzero
        ratios = t2[overlap] / t1[overlap]
        alpha = ratios[0]
        
        # Check all ratios are consistent
        if not np.all(np.abs(ratios - alpha) < 1e-9):
            return 0.0, False
        
        # Check support matches: where t1 is nonzero, t2 should be nonzero (if alpha != 0)
        # and where t2 is nonzero, t1 should be nonzero
        if abs(alpha) > 1e-12:
            if np.any(nonzero2 & ~nonzero1):
                return 0.0, False
            if np.any(nonzero1 & ~nonzero2):
                return 0.0, False
        
        return alpha, True
    
    # Greedy column merging
    r = U_arr.shape[1]
    merged_any = True
    while merged_any and (time.time() - start_time) < DEADLINE:
        merged_any = False
        r = U_arr.shape[1]
        for c1 in range(r):
            if (time.time() - start_time) > DEADLINE:
                break
            for c2 in range(c1 + 1, r):
                alpha, success = can_merge(U_arr, V_arr, W_arr, c1, c2)
                if success:
                    # Merge: scale c1 by (1+alpha) to absorb c2's contribution
                    U_arr[:, c1] *= (1.0 + alpha)
                    U_arr[:, c2] = 0
                    V_arr[:, c2] = 0
                    W_arr[:, c2] = 0
                    merged_any = True
                    break
            if merged_any:
                break
    
    # Remove zero columns
    nonzero_cols = [c for c in range(U_arr.shape[1]) 
                    if np.any(np.abs(U_arr[:, c]) > 1e-12) or 
                       np.any(np.abs(V_arr[:, c]) > 1e-12) or 
                       np.any(np.abs(W_arr[:, c]) > 1e-12)]
    
    U_final = U_arr[:, nonzero_cols]
    V_final = V_arr[:, nonzero_cols]
    W_final = W_arr[:, nonzero_cols]
    
    final_rank = len(nonzero_cols)
    
    # Verify
    valid, err = _verify(U_final.tolist(), V_final.tolist(), W_final.tolist(), n, m, p)
    
    if valid and final_rank < best_rank:
        best_rank = final_rank
        best_result = (U_final.tolist(), V_final.tolist(), W_final.tolist())
    elif not valid:
        # Fall back to schoolbook
        best_result = _schoolbook(n, m, p)
        best_rank = n * m * p
    
    # Check deadline
    if (time.time() - start_time) > DEADLINE:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    # Try additional optimization: look for more complex merges
    # Try to find 3-column merges where the sum of 3 rank-1 tensors equals another
    
    # For now, return the best we found
    U, V, W = best_result
    
    # Convert to list of lists for the interface
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
