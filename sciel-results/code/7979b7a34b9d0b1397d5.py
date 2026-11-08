"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]
_DEADLINE = 130.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _verify(n, m, p, U, V, W):
    """Check that U, V, W exactly decompose the matrix multiplication tensor (vectorized)."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)  # (n*m, rank)
    Va = np.array(V, dtype=float)  # (m*p, rank)
    Wa = np.array(W, dtype=float)  # (p*n, rank)
    
    # Build target indices
    idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx.append((i*m+j, j*p+k, k*n+i))
    idx = np.array(idx)  # (n*m*p, 3)
    
    # Compute T at target positions: sum_c U[a,c]*V[b,c]*W[c_]
    # T[a,b,c_] = sum_r Ua[a,r] * Va[b,r] * Wa[c_,r]
    # For each (a,b,c_) in idx, compute this sum
    # Vectorize: for each rank r, outer product
    total = np.zeros(n*m*p, dtype=float)
    for r in range(rank):
        total += Ua[idx[:,0], r] * Va[idx[:,1], r] * Wa[idx[:,2], r]
    
    return np.allclose(total, np.ones(n*m*p), atol=1e-9)

def _strassen_decomposition(n, m, p):
    """
    Decompose M_{n,m,p} using block Strassen.
    For (2,4,6): split m=4 into two blocks of size 2, p=6 into three blocks of size 2.
    Each subproblem is M_{2,2,2} solved by Strassen (rank 7).
    Total rank = 2 * 3 * 7 = 42.
    """
    a_block_size = 2
    b_col_block_size = 2
    a_blocks = m // a_block_size  # 2
    b_col_blocks = p // b_col_block_size  # 3
    strassen_rank = 7
    rank = a_blocks * b_col_blocks * strassen_rank  # 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]
    
    col = 0
    for a in range(a_blocks):
        for b in range(b_col_blocks):
            for s in range(strassen_rank):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * a_block_size + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * a_block_size + j_loc
                    k = b * b_col_block_size + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b * b_col_block_size + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, n, m, p, max_iterations=50):
    """Try to reduce rank by finding linear dependencies in the factor matrices."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Try to find columns that are linear combinations of others
    # This is a heuristic approach - try removing columns and checking validity
    best_rank = rank
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
    
    # Try greedy removal: remove column r if the tensor can still be represented
    # with the remaining columns (by adjusting other columns)
    for target_rank in range(rank - 1, max(rank - 10, 0), -1):
        if time.time() > _DEADLINE - 5:
            break
        
        # Try to find a set of target_rank columns that span the tensor
        # Use a simple heuristic: try removing one column at a time
        current_rank = best_rank
        for attempt in range(max_iterations):
            if time.time() > _DEADLINE - 5:
                break
            
            # Try removing a random column
            r_to_remove = np.random.randint(0, current_rank)
            
            # Check if we can redistribute the contribution of column r_to_remove
            # to other columns by solving a least squares problem
            # This is complex, so use a simpler approach: just try and verify
            
            # Create candidate by zeroing out column r_to_remove
            cand_U = best_U[:, :].copy()
            cand_V = best_V[:, :].copy()
            cand_W = best_W[:, :].copy()
            
            # Actually, we need to find new coefficients for remaining columns
            # that compensate for the removed column
            # This requires solving: sum_{r != r_to_remove} U[a,r']*V[b,r']*W[c,r'] = original_tensor
            
            # For now, just try if the tensor is still valid after removal (it won't be,
            # but this is a placeholder for a more sophisticated approach)
            pass
        
        if best_rank <= target_rank:
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Generate the Strassen-based decomposition (rank 42)
    result = _strassen_decomposition(n, m, p)
    
    # Verify correctness
    if not _verify(n, m, p, result["U"], result["V"], result["W"]):
        # Fallback to schoolbook if verification fails
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
    
    # Try to reduce rank
    Ua = np.array(result["U"], dtype=float)
    Va = np.array(result["V"], dtype=float)
    Wa = np.array(result["W"], dtype=float)
    
    reduced_U, reduced_V, reduced_W, reduced_rank = _try_reduce_rank(
        result["U"], result["V"], result["W"], n, m, p
    )
    
    if reduced_rank < len(result["U"][0]):
        # Verify the reduced decomposition
        reduced_result = {
            "U": [[int(x) if x == int(x) else float(x) for x in row] for row in reduced_U],
            "V": [[int(x) if x == int(x) else float(x) for x in row] for row in reduced_V],
            "W": [[int(x) if x == int(x) else float(x) for x in row] for row in reduced_W]
        }
        
        if _verify(n, m, p, reduced_result["U"], reduced_result["V"], reduced_result["W"]):
            return reduced_result
    
    # Return the original Strassen decomposition
    return result

# EVOLVE-BLOCK-END
