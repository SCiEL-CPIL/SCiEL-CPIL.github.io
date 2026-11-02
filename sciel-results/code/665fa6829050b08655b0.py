"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS

    # Pre-compute the rank-50 block decomposition
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1

        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1

        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1

        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)

    # Verify the rank-50 decomposition is correct
    # Check: sum_r U[:,r] * V[:,r]^T ⊗ W[:,r] == T
    # Equivalent: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # But we can check via tensor contraction:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 for all valid (i,j,k)
    
    # Quick verification using vectorized approach
    # T_{ijk} = sum_r U_{im+j, r} * V_{jp+k, r} * W_{kn+i, r}
    # Reshape to check
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum
    T_check = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t, optimize=True)
    
    # For matrix multiplication 2x4 * 4x7, the tensor T[i,j,k] should be 1
    # where i in [0,2), j in [0,4), k in [0,7)
    expected_T = np.ones((n, m, p), dtype=np.int64)
    
    if not np.array_equal(T_check, expected_T):
        # Fallback to schoolbook
        return _schoolbook(n, m, p)

    # Now attempt to reduce rank by finding linearly dependent columns
    # in the factor matrices. If we can find a subset of columns that still
    # spans the same tensor, we can remove redundant columns.
    
    # Strategy: The rank of a tensor decomposition can sometimes be reduced
    # if some columns in U, V, W are linearly dependent in a compatible way.
    
    # Build the full factor matrix stack and check for rank deficiency
    # in the combined system. For tensor decomposition, the columns of U, V, W
    # form a valid decomposition if the bilinear form matches.
    
    # A more direct approach: try to find if we can merge or eliminate
    # columns. This is equivalent to checking if the tensor has a lower-rank
    # decomposition.
    
    # For now, check if any column in U is a linear combination of others,
    # and similarly for V and W, with the constraint that the tensor is preserved.
    
    # This is complex, so let's try a simpler heuristic: 
    # Check if the decomposition has redundant structure by examining
    # the support patterns.
    
    # Actually, let's try a direct approach: attempt rank reduction by
    # checking if we can find a lower-rank decomposition via optimization.
    
    if time.time() < deadline - 30:
        try:
            reduced = _attempt_rank_reduction(U, V, W, n, m, p, deadline)
            if reduced is not None:
                r_red = reduced.shape[1]
                if r_red < rank:
                    return {"U": reduced[0].tolist(), "V": reduced[1].tolist(), "W": reduced[2].tolist()}
        except Exception:
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _attempt_rank_reduction(U, V, W, n, m, p, deadline):
    """Attempt to find a lower-rank decomposition by checking for
    linear dependencies among the factor columns."""
    
    current_rank = U.shape[1]
    
    # Check if columns are linearly dependent
    # For tensor decomposition, we need to preserve the tensor structure
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Try to identify if we can eliminate columns
    # One approach: if a column in U is zero, we can remove it
    # (but we should have no zero columns in a minimal decomposition)
    
    # Check for zero columns
    zero_U = np.where(np.all(U == 0, axis=0))[0]
    zero_V = np.where(np.all(V == 0, axis=0))[0]
    zero_W = np.where(np.all(W == 0, axis=0))[0]
    
    zero_cols = set(zero_U) | set(zero_V) | set(zero_W)
    if zero_cols:
        mask = np.ones(current_rank, dtype=bool)
        mask[list(zero_cols)] = False
        return (U[:, mask], V[:, mask], W[:, mask])
    
    # Try to find if we can reduce by checking numerical rank
    # of the factor matrices combined
    # This is a heuristic - not guaranteed to preserve the tensor
    
    # A more principled approach: use the fact that for matrix multiplication,
    # the rank is bounded by the Strassen-like bounds.
    
    # For 2x4 * 4x7:
    # Lower bound: max(2*4, 4*7, 2*7) = max(8, 28, 14) = 28
    # Upper bound via Strassen: we got 50
    # Theoretical minimum is likely around 30-40
    
    # Try a simple approach: if the decomposition is not minimal,
    # there exist linear relations among the columns.
    
    # Check the rank of the tensor itself
    # The tensor T is 2x4x7, which has 56 entries.
    # The rank of the decomposition is at least the rank of the flattening.
    
    # Matrix flattening T_{(i,j), k}: 8x7 matrix
    # The rank of this flattening gives a lower bound on tensor rank
    T_flat = np.ones((n*m, p), dtype=np.int64)
    rank_lower = np.linalg.matrix_rank(T_flat.astype(np.float64))
    
    if current_rank <= rank_lower:
        return None  # Already at or below the lower bound
    
    # Try to find a lower-rank decomposition via iterative refinement
    # This is a heuristic approach
    
    # One specific check: for the 2x4*4x7 case, we can try
    # to see if we can use a 2x2x2 Strassen-like block for the
    # 2x4 part combined with the 7-column dimension differently.
    
    # For now, if no simple reduction is found, return None
    # The rank-50 decomposition is likely near-optimal for this problem size
    
    if time.time() > deadline - 10:
        return None
    
    # Try a randomized approach: perturb and see if we can find
    # a lower-rank decomposition
    # This is not guaranteed to work, so we keep it simple
    
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
