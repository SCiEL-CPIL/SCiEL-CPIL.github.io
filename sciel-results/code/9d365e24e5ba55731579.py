<<<<<<< SEARCH
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

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Generate the Strassen-based decomposition (rank 42)
    result = _strassen_decomposition(n, m, p)
    
    # Verify correctness
    if _verify(n, m, p, result["U"], result["V"], result["W"]):
        return result
    
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

# EVOLVE-BLOCK-END
=======
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

def _try_reduce_rank(n, m, p, U, V, W, max_time=10.0):
    """
    Try to reduce the rank by checking if any rank-1 term can be removed
    and compensated by adjusting others. Uses a greedy approach:
    for each column r, check if removing it still allows the tensor to be
    represented with the remaining columns (by solving for adjustments).
    
    A simpler approach: use the fact that if two rank-1 terms are linearly
    dependent in a certain sense, we can merge them.
    
    Actually, the most practical approach for small ranks: try to find
    a lower-rank decomposition by solving the tensor equation directly
    with rank r' < r, using iterative methods or checking feasibility.
    
    For now, use a simple heuristic: check if any column is a linear
    combination of others in all three matrices simultaneously.
    """
    start = time.time()
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    # Try to find if any column can be removed
    # For each column r, check if U[:,r], V[:,r], W[:,r] are "redundant"
    # This is complex, so use a simpler approach:
    # Check the numerical rank of the tensor's mode-fibrations
    
    # Build the tensor T of shape (n*m, m*p, p*n)
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # We can check if rank can be reduced by looking at the tensor's
    # slice matrices
    
    # For a rank-r decomposition, the tensor lies in the span of r rank-1 tensors.
    # To check if rank r' < r suffices, we'd need to solve a non-linear system.
    
    # Practical approach: use the "rank profile" method
    # The rank of the matrix multiplication tensor M_{n,m,p} is known to be
    # at most n*m*p (schoolbook) and at least something.
    # For 2x4x6, the known upper bound is 42 (Strassen block) or potentially lower.
    
    # Let's try a different approach: use random projections to estimate
    # the minimal rank and then attempt to construct a decomposition.
    
    # Actually, let's try the simplest possible reduction:
    # Check if any pair of columns are linearly dependent across all three matrices.
    for r in range(rank - 1, -1, -1):
        if time.time() - start > max_time:
            break
        
        # Check if column r can be expressed as a linear combination
        # of the other columns in U, V, and W simultaneously.
        # This requires: U[:,r] = U[:,others] @ x
        #                 V[:,r] = V[:,others] @ y  
        #                 W[:,r] = W[:,others] @ z
        # for some x, y, z. This is generally not possible unless
        # the columns are truly redundant.
        
        # A simpler check: if U[:,r] is in the span of U[:,others],
        # AND V[:,r] is in the span of V[:,others],
        # AND W[:,r] is in the span of W[:,others],
        # then we might be able to remove column r.
        
        U_others = np.delete(Ua, r, axis=1)
        V_others = np.delete(Va, r, axis=1)
        W_others = np.delete(Wa, r, axis=1)
        
        # Check if U[:,r] is in the column space of U_others
        try:
            # Use least squares to see if U[:,r] is in span(U_others)
            U_others_rank = np.linalg.matrix_rank(U_others)
            U_full_rank = np.linalg.matrix_rank(Ua)
            if U_full_rank <= U_others_rank:
                # U[:,r] is in the span
                V_others_rank = np.linalg.matrix_rank(V_others)
                V_full_rank = np.linalg.matrix_rank(Va)
                if V_full_rank <= V_others_rank:
                    W_others_rank = np.linalg.matrix_rank(W_others)
                    W_full_rank = np.linalg.matrix_rank(Wa)
                    if W_full_rank <= W_others_rank:
                        # All three matrices have their column r in the span of others
                        # Try to remove it and verify
                        U_new = np.delete(Ua, r, axis=1)
                        V_new = np.delete(Va, r, axis=1)
                        W_new = np.delete(Wa, r, axis=1)
                        
                        # Verify the decomposition still works
                        if _verify(n, m, p, U_new, V_new, W_new):
                            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
        except np.linalg.LinAlgError:
            pass
    
    return None

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
    
    # Try to reduce the rank
    reduced = _try_reduce_rank(n, m, p, result["U"], result["V"], result["W"], 
                               max_time=max(0, _DEADLINE - (time.time() - start) - 2.0))
    if reduced is not None:
        return reduced
    
    return result

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
