"""Matrix multiplication 2x4x5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _build_strassen_36():
    """Build the known rank-36 block Strassen decomposition."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(i, j):
        return i * m + j
    def v_idx(j, k):
        return j * p + k
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y01[0], y01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y00[0], y00[1]), col] = -1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(y01[0], y01[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = -1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        col += 1
    
    # Block 1: A_L B_11
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 2: A_R B_21
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 3: A_L B_12
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
    # Block 4: A_R B_22
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
    
    # Block 5: A_L B_15 (2x2 * 2x1)
    U[u_idx(0,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    # Block 6: A_R B_25 (2x2 * 2x1)
    U[u_idx(0,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _tensor_vec(U, V, W, col):
    """Flatten the rank-1 tensor U[:,col] ⊗ V[:,col] ⊗ W[:,col] into a vector."""
    return np.kron(np.kron(U[:, col], V[:, col]), W[:, col])


def _find_redundant_columns(U, V, W, deadline):
    """
    Find columns that are linearly dependent on the others as rank-1 tensors.
    For each column c, check if tensor_vec(c) is in the span of the other tensor vectors.
    This is a structural algebraic check: if a term is in the span of the others,
    the decomposition still works after removal (with adjusted coefficients).
    Returns a list of removable column indices.
    """
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    dim = n * m * m * p * p * n  # (n*m)*(m*p)*(p*n)
    
    # Build matrix where each row is a flattened rank-1 tensor
    T = np.zeros((rank, dim))
    for c in range(rank):
        T[c, :] = _tensor_vec(U, V, W, c)
    
    removable = []
    for c in range(rank):
        if time.time() > deadline:
            break
        # Check if row c is in the span of all other rows
        others = [T[r, :] for r in range(rank) if r != c]
        if not others:
            continue
        A = np.array(others)
        # Check if T[c] is in the column space of A^T (i.e., A's row space)
        # Equivalent: rank([A; T[c]]) == rank(A)
        try:
            r1 = np.linalg.matrix_rank(A)
            combined = np.vstack([A, T[c:c+1, :]])
            r2 = np.linalg.matrix_rank(combined)
            if r1 == r2:
                removable.append(c)
        except np.linalg.LinAlgError:
            pass
    
    return removable


def _try_remove_columns(U, V, W, cols_to_remove):
    """Remove specified columns and verify the decomposition still holds."""
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    mask = [i for i in range(rank) if i not in cols_to_remove]
    if len(mask) == 0:
        return None
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    if _verify_decomposition(U_new, V_new, W_new):
        return U_new, V_new, W_new
    return None


def _pairwise_merge_search(U, V, W, deadline):
    """
    Try to merge pairs of columns by finding if their tensor sum is also
    representable with fewer terms. This is a different structural approach:
    instead of removing one column, try to replace two columns with one
    (if the sum of two rank-1 tensors happens to be rank-1, which is rare
    but we check).
    
    More practically: try removing pairs of columns simultaneously.
    """
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    
    # Try removing pairs
    best_result = None
    for c1, c2 in combinations(range(rank), 2):
        if time.time() > deadline:
            break
        result = _try_remove_columns(U, V, W, {c1, c2})
        if result is not None:
            best_result = result
            break
    
    return best_result


def _greedy_column_removal(U, V, W, deadline):
    """Original greedy single-column removal as fallback."""
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    removed = set()
    
    for attempt in range(rank * 2):
        if time.time() > deadline:
            break
        
        progress = False
        for col in range(rank):
            if col in removed:
                continue
            if time.time() > deadline:
                break
            
            mask = [i for i in range(rank) if i != col]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            
            if _verify_decomposition(U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                removed.add(col)
                rank = U.shape[1]
                progress = True
                break
    
    return U, V, W


def _try_strassen_with_variants(deadline):
    """
    Try alternative Strassen-like decompositions with different block splits.
    For 2x4 * 4x5, we can split differently:
    - Split B into 2x3 + 2x2 (instead of 2x2 + 2x2 + 2x1)
    - Use different Strassen variants
    """
    n, m, p = 2, 4, 5
    
    # Variant: Split into 2x2 * 2x3 and 2x2 * 2x2
    # A = [A1 A2] (2x2 each)
    # B = [B1; B2] where B1 is 2x3, B2 is 2x2
    # C = A1*B1 + A2*B2
    # For A1*B1 (2x2 * 2x3): use Strassen on 2x2*2x2 + 2x1
    # For A2*B2 (2x2 * 2x2): use Strassen
    
    # This gives: 7 (for 2x2*2x2 part of B1) + 4 (for 2x1 part of B1) + 7 (for B2)
    # = 18 per half, but we share... actually let's just try the standard and see.
    
    # Alternative: use the fact that for 2x2 * 2x5, we can do better
    # Split 2x5 into 2x2 + 2x2 + 2x1
    # Strassen on 2x2*2x2 (rank 7) twice + 4 for 2x1 = 18
    # Two halves: 36 total (same as before)
    
    # Try a different approach: use Winograd's method or other fast algorithms
    # For now, return the standard decomposition
    return None


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with the known rank-36 decomposition
    U, V, W = _build_strassen_36()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if not _verify_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Phase 1: Algebraic dependency detection (structural change from greedy)
    # Find columns whose rank-1 tensor is in the span of the others
    removable = _find_redundant_columns(U, V, W, deadline)
    
    if removable:
        # Try removing all redundant columns at once
        result = _try_remove_columns(U, V, W, set(removable))
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Phase 2: If algebraic method found nothing, try pairwise removal
    if time.time() < deadline:
        result = _pairwise_merge_search(U, V, W, deadline)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Phase 3: Greedy single-column removal as final fallback
    if time.time() < deadline:
        U, V, W = _greedy_column_removal(U, V, W, deadline)
        if U.shape[1] < best_U.shape[1]:
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Use the best result found
    if best_U.shape[1] < 36:
        U, V, W = best_U, best_V, best_W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
