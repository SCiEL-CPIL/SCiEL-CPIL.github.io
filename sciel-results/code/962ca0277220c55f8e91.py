"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]
_DEADLINE_SECONDS = 115  # safety margin before 120s kill

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return (U, V, W) lists of lists, rank n*m*p."""
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


def _strassen_2x2():
    """Strassen's 7-product algorithm for 2x2x2."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, -1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, -1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, -1, 0, 0, 1, 0]
    ]
    return U, V, W


def _embed_subproblem(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                      i_offset, j_offset, k_offset, n, m, p):
    """Embed a subproblem decomposition into the full space."""
    rank_sub = len(U_sub[0])
    U_full = [[0] * rank_sub for _ in range(n * m)]
    V_full = [[0] * rank_sub for _ in range(m * p)]
    W_full = [[0] * rank_sub for _ in range(p * n)]
    
    for local_i in range(n_sub):
        for local_j in range(m_sub):
            full_i = i_offset + local_i
            full_j = j_offset + local_j
            local_idx = local_i * m_sub + local_j
            full_idx = full_i * m + full_j
            U_full[full_idx] = U_sub[local_i * m_sub + local_j]
    
    for local_j in range(m_sub):
        for local_k in range(p_sub):
            full_j = j_offset + local_j
            full_k = k_offset + local_k
            local_idx = local_j * p_sub + local_k
            full_idx = full_j * p + full_k
            V_full[full_idx] = V_sub[local_j * p_sub + local_k]
    
    for local_k in range(p_sub):
        for local_i in range(n_sub):
            full_k = k_offset + local_k
            full_i = i_offset + local_i
            local_idx = local_k * n_sub + local_i
            full_idx = full_k * n + full_i
            W_full[full_idx] = W_sub[local_k * n_sub + local_i]
    
    return U_full, V_full, W_full


def _concatenate_decompositions(U1, V1, W1, U2, V2, W2, n, m, p):
    """Concatenate two decompositions horizontally."""
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 + r2
    U = [U1[i] + U2[i] for i in range(n * m)]
    V = [V1[i] + V2[i] for i in range(m * p)]
    W = [W1[i] + W2[i] for i in range(p * n)]
    return U, V, W


def _construct_4x5x5_optimized():
    """Build 4x5x5 decomposition with block splits. Rank 92."""
    n, m, p = 4, 5, 5
    
    # Part 1: A[:, 0:2] * B[0:2, :]
    U_1a, V_1a, W_1a = _strassen_2x2()
    U_1a_full, V_1a_full, W_1a_full = _embed_subproblem(
        U_1a, V_1a, W_1a, 2, 2, 2, 0, 0, 0, n, m, p)
    
    U_1b1, V_1b1, W_1b1 = _strassen_2x2()
    U_1b1_full, V_1b1_full, W_1b1_full = _embed_subproblem(
        U_1b1, V_1b1, W_1b1, 2, 2, 2, 0, 0, 2, n, m, p)
    U_1b2, V_1b2, W_1b2 = _schoolbook(2, 2, 1)
    U_1b2_full, V_1b2_full, W_1b2_full = _embed_subproblem(
        U_1b2, V_1b2, W_1b2, 2, 2, 1, 0, 0, 4, n, m, p)
    U_1b_full, V_1b_full, W_1b_full = _concatenate_decompositions(
        U_1b1_full, V_1b1_full, W_1b1_full, U_1b2_full, V_1b2_full, W_1b2_full, n, m, p)
    
    U_1c, V_1c, W_1c = _strassen_2x2()
    U_1c_full, V_1c_full, W_1c_full = _embed_subproblem(
        U_1c, V_1c, W_1c, 2, 2, 2, 2, 0, 0, n, m, p)
    
    U_1d1, V_1d1, W_1d1 = _strassen_2x2()
    U_1d1_full, V_1d1_full, W_1d1_full = _embed_subproblem(
        U_1d1, V_1d1, W_1d1, 2, 2, 2, 2, 0, 2, n, m, p)
    U_1d2, V_1d2, W_1d2 = _schoolbook(2, 2, 1)
    U_1d2_full, V_1d2_full, W_1d2_full = _embed_subproblem(
        U_1d2, V_1d2, W_1d2, 2, 2, 1, 2, 0, 4, n, m, p)
    U_1d_full, V_1d_full, W_1d_full = _concatenate_decompositions(
        U_1d1_full, V_1d1_full, W_1d1_full, U_1d2_full, V_1d2_full, W_1d2_full, n, m, p)
    
    U_part1, V_part1, W_part1 = U_1a_full, V_1a_full, W_1a_full
    U_part1, V_part1, W_part1 = _concatenate_decompositions(
        U_part1, V_part1, W_part1, U_1b_full, V_1b_full, W_1b_full, n, m, p)
    U_part1, V_part1, W_part1 = _concatenate_decompositions(
        U_part1, V_part1, W_part1, U_1c_full, V_1c_full, W_1c_full, n, m, p)
    U_part1, V_part1, W_part1 = _concatenate_decompositions(
        U_part1, V_part1, W_part1, U_1d_full, V_1d_full, W_1d_full, n, m, p)
    
    # Part 2: A[:, 2:5] * B[2:5, :]
    U_2a, V_2a, W_2a = _schoolbook(4, 1, 5)
    U_2a_full, V_2a_full, W_2a_full = _embed_subproblem(
        U_2a, V_2a, W_2a, 4, 1, 5, 0, 2, 0, n, m, p)
    
    U_2b1a, V_2b1a, W_2b1a = _strassen_2x2()
    U_2b1a_full, V_2b1a_full, W_2b1a_full = _embed_subproblem(
        U_2b1a, V_2b1a, W_2b1a, 2, 2, 2, 0, 3, 0, n, m, p)
    
    U_2b1b1, V_2b1b1, W_2b1b1 = _strassen_2x2()
    U_2b1b1_full, V_2b1b1_full, W_2b1b1_full = _embed_subproblem(
        U_2b1b1, V_2b1b1, W_2b1b1, 2, 2, 2, 0, 3, 2, n, m, p)
    U_2b1b2, V_2b1b2, W_2b1b2 = _schoolbook(2, 2, 1)
    U_2b1b2_full, V_2b1b2_full, W_2b1b2_full = _embed_subproblem(
        U_2b1b2, V_2b1b2, W_2b1b2, 2, 2, 1, 0, 3, 4, n, m, p)
    U_2b1b_full, V_2b1b_full, W_2b1b_full = _concatenate_decompositions(
        U_2b1b1_full, V_2b1b1_full, W_2b1b1_full, U_2b1b2_full, V_2b1b2_full, W_2b1b2_full, n, m, p)
    
    U_2b2a, V_2b2a, W_2b2a = _strassen_2x2()
    U_2b2a_full, V_2b2a_full, W_2b2a_full = _embed_subproblem(
        U_2b2a, V_2b2a, W_2b2a, 2, 2, 2, 2, 3, 0, n, m, p)
    
    U_2b2b1, V_2b2b1, W_2b2b1 = _strassen_2x2()
    U_2b2b1_full, V_2b2b1_full, W_2b2b1_full = _embed_subproblem(
        U_2b2b1, V_2b2b1, W_2b2b1, 2, 2, 2, 2, 3, 2, n, m, p)
    U_2b2b2, V_2b2b2, W_2b2b2 = _schoolbook(2, 2, 1)
    U_2b2b2_full, V_2b2b2_full, W_2b2b2_full = _embed_subproblem(
        U_2b2b2, V_2b2b2, W_2b2b2, 2, 2, 1, 2, 3, 4, n, m, p)
    U_2b2b_full, V_2b2b_full, W_2b2b_full = _concatenate_decompositions(
        U_2b2b1_full, V_2b2b1_full, W_2b2b1_full, U_2b2b2_full, V_2b2b2_full, W_2b2b2_full, n, m, p)
    
    U_2b_full, V_2b_full, W_2b_full = U_2b1a_full, V_2b1a_full, W_2b1a_full
    U_2b_full, V_2b_full, W_2b_full = _concatenate_decompositions(
        U_2b_full, V_2b_full, W_2b_full, U_2b1b_full, V_2b1b_full, W_2b1b_full, n, m, p)
    U_2b_full, V_2b_full, W_2b_full = _concatenate_decompositions(
        U_2b_full, V_2b_full, W_2b_full, U_2b2a_full, V_2b2a_full, W_2b2a_full, n, m, p)
    U_2b_full, V_2b_full, W_2b_full = _concatenate_decompositions(
        U_2b_full, V_2b_full, W_2b_full, U_2b2b_full, V_2b2b_full, W_2b2b_full, n, m, p)
    
    U_part2, V_part2, W_part2 = _concatenate_decompositions(
        U_2a_full, V_2a_full, W_2a_full, U_2b_full, V_2b_full, W_2b_full, n, m, p)
    
    U, V, W = _concatenate_decompositions(
        U_part1, V_part1, W_part1, U_part2, V_part2, W_part2, n, m, p)
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(n):
                    s = float(np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+r]))
                    expected = 1.0 if (i == r and j == k) else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True


def _reduce_rank(U, V, W, n, m, p, deadline):
    """Try to reduce rank by finding linear dependencies among factor vectors.
    
    If column c of (U,V,W) is a linear combination of other columns,
    we can eliminate it. We check each column against the span of others.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Stack all factor vectors: each column c gives a vector in R^{n*m + m*p + p*n}
    # But we need to be careful: a column is redundant if there exist coefficients
    # such that for all (i,j,k,r): sum_{c'} alpha_{c'} * U[i*m+j,c'] * V[j*p+k,c'] * W[k*n+r,c']
    # equals the contribution of column c.
    # 
    # Simpler approach: check if the set of "rank-1 tensors" {u_c ⊗ v_c ⊗ w_c} is linearly
    # dependent. If so, we can drop one.
    #
    # Even simpler: for small rank reductions, check if any column is exactly zero
    # or if two columns are identical.
    
    # First pass: remove zero columns
    keep = []
    for c in range(rank):
        if U_np[:, c].any() or V_np[:, c].any() or W_np[:, c].any():
            keep.append(c)
    
    if len(keep) < rank:
        U_np = U_np[:, keep]
        V_np = V_np[:, keep]
        W_np = W_np[:, keep]
        rank = len(keep)
    
    # Second pass: check for duplicate columns (up to sign)
    # Build a hash for each column triple
    seen = {}
    keep2 = []
    for c in range(rank):
        u_col = tuple(U_np[:, c])
        v_col = tuple(V_np[:, c])
        w_col = tuple(W_np[:, c])
        key = (u_col, v_col, w_col)
        neg_key = (tuple(-x for x in u_col), tuple(-x for x in v_col), tuple(-x for x in w_col))
        if key in seen or neg_key in seen:
            continue
        seen[key] = True
        keep2.append(c)
    
    if len(keep2) < rank:
        U_np = U_np[:, keep2]
        V_np = V_np[:, keep2]
        W_np = W_np[:, keep2]
        rank = len(keep2)
    
    # Convert back to lists
    U_out = U_np.tolist()
    V_out = V_np.tolist()
    W_out = W_np.tolist()
    
    return U_out, V_out, W_out


def construct():
    """Construct the rank-minimal exact tensor decomposition for 4x5x5 matrix mult."""
    t0 = time.time()
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # Try the improved block decomposition
    try:
        U, V, W = _construct_4x5x5_optimized()
        
        # Verify correctness
        if _verify_decomposition(U, V, W, n, m, p):
            # Try to reduce rank
            if time.time() - t0 < _DEADLINE_SECONDS - 10:
                U, V, W = _reduce_rank(U, V, W, n, m, p, _DEADLINE_SECONDS)
                
                # Re-verify after reduction
                if _verify_decomposition(U, V, W, n, m, p):
                    if time.time() - t0 < _DEADLINE_SECONDS - 5:
                        return {"U": U, "V": V, "W": W}
            else:
                if time.time() - t0 < _DEADLINE_SECONDS - 5:
                    return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
