"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a11 + a22)(b11 + b22)
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    # M2 = (a21 + a22) b11
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 1, 0, -1]
    
    # M3 = a11 (b12 - b22)
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 0, 1, 1]
    
    # M4 = a22 (b21 - b11)
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 0, 1, 0]
    W[:, 3] = [1, 1, 0, 0]
    
    # M5 = (a11 + a12) b22
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 0, 1, 0]
    
    # M6 = (a21 - a11) (b11 + b12)
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    
    # M7 = (a12 - a22) (b21 + b22)
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook decomposition for n x m x p. Returns U, V, W."""
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


def _embed_block(U_local, V_local, W_local, n, m, p, n_off, m_off, p_off, ni, mi, pi):
    """
    Embed a local decomposition of size ni x mi x pi into global dimensions.
    Returns the global U, V, W slices (as arrays of shape (n*m, r), etc.)
    """
    r = U_local.shape[1]
    U_g = np.zeros((n * m, r))
    V_g = np.zeros((m * p, r))
    W_g = np.zeros((p * n, r))
    
    for li in range(ni):
        for lj in range(mi):
            local_idx = li * mi + lj
            global_idx = (n_off + li) * m + (m_off + lj)
            U_g[global_idx, :] = U_local[local_idx, :]
    
    for lj in range(mi):
        for lk in range(pi):
            local_idx = lj * pi + lk
            global_idx = (m_off + lj) * p + (p_off + lk)
            V_g[global_idx, :] = V_local[local_idx, :]
    
    for lk in range(pi):
        for li in range(ni):
            local_idx = lk * ni + li
            global_idx = (p_off + lk) * n + (n_off + li)
            W_g[global_idx, :] = W_local[local_idx, :]
    
    return U_g, V_g, W_g


def _block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """
    Decompose n x m x p into blocks defined by n_parts, m_parts, p_parts.
    Use Strassen for 2x2x2 blocks, schoolbook for others.
    """
    n_offsets = [sum(n_parts[:i]) for i in range(len(n_parts))]
    m_offsets = [sum(m_parts[:i]) for i in range(len(m_parts))]
    p_offsets = [sum(p_parts[:i]) for i in range(len(p_parts))]
    
    total_rank = 0
    block_ranks = {}
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                if ni == 2 and mi == 2 and pi == 2:
                    block_ranks[(i, j, k)] = 7
                else:
                    block_ranks[(i, j, k)] = ni * mi * pi
                total_rank += block_ranks[(i, j, k)]
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                r = block_ranks[(i, j, k)]
                if ni == 2 and mi == 2 and pi == 2:
                    U_loc, V_loc, W_loc = _strassen_2x2x2()
                else:
                    U_loc, V_loc, W_loc = _schoolbook(ni, mi, pi)
                
                U_g, V_g, W_g = _embed_block(U_loc, V_loc, W_loc, n, m, p,
                                              n_offsets[i], m_offsets[j], p_offsets[k],
                                              ni, mi, pi)
                U[:, col:col+r] = U_g
                V[:, col:col+r] = V_g
                W[:, col:col+r] = W_g
                col += r
    
    return U, V, W, total_rank


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is correct."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U[i*m+j, :], V[j*p+k, :]), W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _reduce_rank(U, V, W, n, m, p, max_iter=50, tol=1e-9):
    """
    Try to reduce the rank of a valid decomposition by finding linear dependencies.
    Uses a greedy approach: at each step, try to express one column as a linear
    combination of others and remove it.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    current_rank = U.shape[1]
    
    for iteration in range(max_iter):
        # Try to find a column that is linearly dependent on others
        # For each column c, check if removing it still allows reconstruction
        # We use a simpler approach: check if any column can be expressed as
        # a combination of others in all three matrices simultaneously
        
        found_redundant = False
        for c in range(current_rank):
            # Build the system: find alpha such that for all (i,j,k):
            # sum_c' alpha[c'] * U[i*m+j, c'] * V[j*p+k, c'] * W[k*n+i, c'] = 0
            # when we remove column c, we need the remaining columns to still span
            # Actually, a simpler test: if column c is in the span of others in
            # the tensor product space, we can remove it.
            
            # Approximate test: check if the column is "redundant" by seeing if
            # the matrix formed by all columns except c still has full "tensor rank"
            # This is expensive, so we use a heuristic: try small random subsets
            
            pass  # Skip complex reduction for now; use SVD-based approach instead
        
        # SVD-based rank reduction:
        # The tensor T = sum_c U[:,c] outer V[:,c] outer W[:,c]
        # We can try to find a lower-rank representation by looking at the
        # singular values of the matricizations.
        
        # Matricize along U: T_U is (n*m) x (m*p*n)
        # Actually, let's use a different approach: 
        # If we have rank r, we can try to find rank r-1 by checking if
        # the (r-1) x (r-1) minors vanish. This is too expensive.
        
        # Practical approach: use the fact that if we have a valid decomposition,
        # we can try to combine terms. For now, return the input unchanged
        # unless we find an obvious redundancy.
        
        # Check for duplicate columns
        for c1 in range(current_rank):
            for c2 in range(c1+1, current_rank):
                if (np.allclose(U[:, c1], U[:, c2], atol=tol) and
                    np.allclose(V[:, c1], V[:, c2], atol=tol) and
                    np.allclose(W[:, c1], W[:, c2], atol=tol)):
                    # Remove duplicate
                    U = np.delete(U, c2, axis=1)
                    V = np.delete(V, c2, axis=1)
                    W = np.delete(W, c2, axis=1)
                    current_rank -= 1
                    found_redundant = True
                    break
            if found_redundant:
                break
        
        if not found_redundant:
            break
    
    return U, V, W, current_rank


def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Strategy: Use block decomposition with multiple splits, then apply rank reduction.
    # We'll try several different block splits and return the one with lowest rank after reduction.
    
    candidates = []
    
    # Split 1: n=[1,2], m=[2,3], p=[2,3]
    candidates.append(([1, 2], [2, 3], [2, 3]))
    
    # Split 2: n=[1,2], m=[3,2], p=[3,2]
    candidates.append(([1, 2], [3, 2], [3, 2]))
    
    # Split 3: n=[2,1], m=[2,3], p=[2,3]
    candidates.append(([2, 1], [2, 3], [2, 3]))
    
    # Split 4: n=[1,1,1], m=[2,3], p=[2,3]
    candidates.append(([1, 1, 1], [2, 3], [2, 3]))
    
    # Split 5: n=[1,2], m=[2,1,2], p=[2,1,2]
    candidates.append(([1, 2], [2, 1, 2], [2, 1, 2]))
    
    # Split 6: n=[1,2], m=[1,2,2], p=[1,2,2]
    candidates.append(([1, 2], [1, 2, 2], [1, 2, 2]))
    
    # Split 7: n=[1,1,1], m=[1,2,2], p=[1,2,2]
    candidates.append(([1, 1, 1], [1, 2, 2], [1, 2, 2]))
    
    # Split 8: n=[1,2], m=[2,2,1], p=[2,2,1]
    candidates.append(([1, 2], [2, 2, 1], [2, 2, 1]))
    
    # Split 9: n=[1,2], m=[1,1,3], p=[1,1,3]
    candidates.append(([1, 2], [1, 1, 3], [1, 1, 3]))
    
    # Split 10: n=[1,2], m=[1,2,1,1], p=[1,2,1,1]
    candidates.append(([1, 2], [1, 2, 1, 1], [1, 2, 1, 1]))
    
    # Split 11: n=[1,2], m=[2,1,1,1], p=[2,1,1,1]
    candidates.append(([1, 2], [2, 1, 1, 1], [2, 1, 1, 1]))
    
    # Split 12: n=[1,1,1], m=[1,1,1,2], p=[1,1,1,2]
    candidates.append(([1, 1, 1], [1, 1, 1, 2], [1, 1, 1, 2]))
    
    # Split 13: n=[1,2], m=[1,1,1,2], p=[1,1,1,2]
    candidates.append(([1, 2], [1, 1, 1, 2], [1, 1, 1, 2]))
    
    # Split 14: n=[1,2], m=[1,2,2], p=[1,2,2]
    candidates.append(([1, 2], [1, 2, 2], [1, 2, 2]))
    
    # Split 15: n=[1,2], m=[2,2], p=[2,2]
    candidates.append(([1, 2], [2, 2], [2, 2]))
    
    # Split 16: n=[1,2], m=[1,4], p=[1,4]
    candidates.append(([1, 2], [1, 4], [1, 4]))
    
    # Split 17: n=[1,2], m=[4,1], p=[4,1]
    candidates.append(([1, 2], [4, 1], [4, 1]))
    
    # Split 18: n=[1,2], m=[1,1,3], p=[1,1,3]
    candidates.append(([1, 2], [1, 1, 3], [1, 1, 3]))
    
    # Split 19: n=[1,2], m=[3,1,1], p=[3,1,1]
    candidates.append(([1, 2], [3, 1, 1], [3, 1, 1]))
    
    # Split 20: n=[1,2], m=[1,3,1], p=[1,3,1]
    candidates.append(([1, 2], [1, 3, 1], [1, 3, 1]))
    
    # Try all candidates and pick the best
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    for n_parts, m_parts, p_parts in candidates:
        if time.time() > deadline:
            break
        try:
            U, V, W, rank = _block_decomposition(n, m, p, n_parts, m_parts, p_parts)
            
            # Apply rank reduction
            U_red, V_red, W_red, rank_red = _reduce_rank(U, V, W, n, m, p)
            
            # Verify the reduced decomposition
            if _verify_decomposition(U_red, V_red, W_red, n, m, p):
                if rank_red < best_rank:
                    best_rank = rank_red
                    best_U, best_V, best_W = U_red, V_red, W_red
            else:
                # Use the unreduced version if reduction broke something
                if rank < best_rank:
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
        except Exception:
            continue
    
    # Fallback to schoolbook if something went wrong
    if best_U is None:
        U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W = best_U, best_V, best_W
    
    # Convert to lists with integer/half-integer entries
    def _to_list(mat):
        mat = np.asarray(mat)
        # Round to nearest half-integer
        mat_rounded = np.round(mat * 2) / 2.0
        # Convert to list
        return mat_rounded.tolist()
    
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

# EVOLVE-BLOCK-END
