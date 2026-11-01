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


def _verify_fast(U, V, W):
    """Faster verification using numpy."""
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    # Compute C[i,k] = sum_j A[i,j] B[j,k] via the decomposition
    # C[i,k] = sum_r U[i*m+?, r] ... wait, the tensor is T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    # For each (i,k), C[i,k] = sum_j T[i,j,k]
    # Let's just compute the full tensor and check
    # Actually, let's use the property: for the matmul tensor, T[i,j,k] should be 1 if we're computing C[i,k] from A[i,j]B[j,k]
    # No wait, the tensor IS the multiplication. T[i,j,k] = delta(i,j,k) in the sense that 
    # C[i,k] = sum_j A[i,j] B[j,k]. The tensor decomposition gives the bilinear form.
    # So we need: for all i,j,k, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 if this is the "multiplication tensor"
    # But actually the multiplication tensor has T[i,j,k] = 1 for all i,j,k? No.
    # The tensor is T such that C = A*B means C[i,k] = sum_j A[i,j] B[j,k]
    # As a trilinear form: sum_{i,j,k} C[i,k] A[i,j] B[j,k] ... no.
    # The standard tensor: T[i,j,k] = 1 for all valid i,j,k, and C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # So yes, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Fast check: compute all entries
    # Build 3D array via broadcasting
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This is complex to vectorize directly. Let's do it in a semi-vectorized way.
    
    # For each r, the contribution is outer(U[:,r], V[:,r]) reshaped and multiplied with W[:,r]
    # Let's just do the loop but with numpy for the inner sum
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _find_redundant_columns_fast(U, V, W, deadline):
    """
    Find columns whose rank-1 tensor is in the span of the others.
    Uses a more efficient approach: build the tensor matrix once, then check.
    """
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    
    # Instead of flattening into a huge vector, we can check linear dependence
    # of the rank-1 tensors more efficiently.
    # Two rank-1 tensors u⊗v⊗w and u'⊗v'⊗w' are linearly dependent iff
    # u is proportional to u', v to v', and w to w' (with consistent scalar).
    # But for general linear dependence (one in span of others), we need the full check.
    
    # Alternative: use the fact that if column c is redundant, then for all (i,j,k):
    # U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = sum_{c' != c} alpha_{c'} * U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c']
    # This is a system of linear equations in alpha. We can solve it.
    
    # Build the matrix of equations: for each (i,j,k), we have one equation.
    # num_equations = n*m*p = 40
    # num_unknowns = rank - 1
    
    removable = []
    for c in range(rank):
        if time.time() > deadline:
            break
        
        # Build RHS and coefficient matrix
        rhs = np.zeros(n*m*p)
        A_mat = np.zeros((n*m*p, rank - 1))
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i*m*p + j*p + k
                    rhs[eq_idx] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    
                    for c2 in range(rank):
                        if c2 == c:
                            continue
                        A_mat[eq_idx, col_idx] = U[i*m+j, c2] * V[j*p+k, c2] * W[k*n+i, c2]
                        col_idx += 1
        
        # Check if rhs is in the column space of A_mat
        # i.e., rank([A_mat, rhs]) == rank(A_mat)
        try:
            r1 = np.linalg.matrix_rank(A_mat)
            combined = np.hstack([A_mat, rhs.reshape(-1, 1)])
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
    if _verify_fast(U_new, V_new, W_new):
        return U_new, V_new, W_new
    return None


def _greedy_column_removal(U, V, W, deadline):
    """Greedy single-column removal."""
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    
    for attempt in range(rank * 2):
        if time.time() > deadline:
            break
        
        progress = False
        for col in range(rank):
            if time.time() > deadline:
                break
            
            mask = [i for i in range(rank) if i != col]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            
            if _verify_fast(U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                rank = U.shape[1]
                progress = True
                break
        
        if not progress:
            break
    
    return U, V, W


def _try_alternative_splits(deadline):
    """
    Try alternative block splits that might yield lower rank.
    
    Key insight: For 2x4 * 4x5, we can split B differently.
    Instead of 2x2+2x2+2x1, try 2x3+2x2 or other splits.
    
    For 2x2 * 2x3: split 2x3 into 2x2+2x1, use Strassen(7) + 4 = 11
    For 2x2 * 2x2: Strassen = 7
    Total per A-half: 11 + 7 = 18
    Two halves: 36 (same)
    
    But what if we use a different algorithm for 2x2 * 2x3?
    The rank of 2x2*2x3 is at most 2*2*3=12 (schoolbook) or 7+4=11 (Strassen+schoolbook).
    Actually, the rank of 2x2*2x3 is known to be at most 11, and at least 10.
    
    Let's try: split B into 2x2 + 2x2 + 2x1 but use a shared Strassen structure.
    """
    n, m, p = 2, 4, 5
    
    # Try: use the fact that for the two 2x1 blocks, we can combine them.
    # A_L * B_15 + A_R * B_25 = [A_L A_R] * [B_15; B_25]
    # This is a 2x4 * 4x1 multiplication.
    # Rank of 2x4*4x1: at most 2*4*1=8, but can be done in rank 4?
    # C[i,0] = sum_j A[i,j] B[j,0] for i=0,1
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0] + A[0,3]B[3,0]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0] + A[1,3]B[3,0]
    # This is 8 terms schoolbook. Can we do better?
    # It's a bilinear map from (2x4) x (4x1) to (2x1).
    # Rank is at most 8, and I believe it's exactly 8 for this case.
    
    # Let's try a different approach: use the 2x4*4x5 directly with a 
    # non-block decomposition. This is harder to construct manually.
    
    # For now, return None to indicate no improvement found.
    return None


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with the known rank-36 decomposition
    U, V, W = _build_strassen_36()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    if not _verify_fast(U, V, W):
        # Fallback to original verification
        if not _verify_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Phase 1: Algebraic dependency detection
    if time.time() < deadline:
        removable = _find_redundant_columns_fast(U, V, W, deadline)
        if removable:
            result = _try_remove_columns(U, V, W, set(removable))
            if result is not None:
                U, V, W = result
                if U.shape[1] < best_rank:
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = U.shape[1]
    
    # Phase 2: Greedy single-column removal
    if time.time() < deadline:
        U, V, W = _greedy_column_removal(U, V, W, deadline)
        if U.shape[1] < best_rank:
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = U.shape[1]
    
    # Phase 3: Try alternative splits (future improvement)
    if time.time() < deadline:
        alt = _try_alternative_splits(deadline)
        if alt is not None:
            U_alt, V_alt, W_alt = alt
            if _verify_fast(U_alt, V_alt, W_alt) and U_alt.shape[1] < best_rank:
                best_U, best_V, best_W = U_alt.copy(), V_alt.copy(), W_alt.copy()
                best_rank = U_alt.shape[1]
    
    # Use the best result found
    U, V, W = best_U, best_V, best_W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
