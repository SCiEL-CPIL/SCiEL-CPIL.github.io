import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Build the rank-50 decomposition
    rank_init = 50
    U = np.zeros((n * m, rank_init), dtype=np.int64)
    V = np.zeros((m * p, rank_init), dtype=np.int64)
    W = np.zeros((p * n, rank_init), dtype=np.int64)
    
    col = 0
    
    def emit_strassen(i0, j0, k0):
        nonlocal col
        # 7 Strassen terms for 2x2 x 2x2
        U[i0*m + j0, col] = 1
        U[(i0+1)*m + (j0+1), col] = 1
        V[j0*p + k0, col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        W[k0*n + i0, col] = 1
        W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        U[(i0+1)*m + j0, col] = 1
        U[(i0+1)*m + (j0+1), col] = 1
        V[j0*p + k0, col] = 1
        W[k0*n + (i0+1), col] = 1
        W[(k0+1)*n + (i0+1), col] = -1
        col += 1
        
        U[i0*m + j0, col] = 1
        V[j0*p + (k0+1), col] = 1
        V[(j0+1)*p + (k0+1), col] = -1
        W[(k0+1)*n + i0, col] = 1
        W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        U[(i0+1)*m + (j0+1), col] = 1
        V[(j0+1)*p + k0, col] = 1
        V[j0*p + k0, col] = -1
        W[k0*n + (i0+1), col] = 1
        W[k0*n + i0, col] = 1
        col += 1
        
        U[i0*m + j0, col] = 1
        U[i0*m + (j0+1), col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        W[k0*n + i0, col] = -1
        W[(k0+1)*n + i0, col] = 1
        col += 1
        
        U[(i0+1)*m + j0, col] = 1
        U[i0*m + j0, col] = -1
        V[j0*p + k0, col] = 1
        V[j0*p + (k0+1), col] = 1
        W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        U[i0*m + (j0+1), col] = 1
        U[(i0+1)*m + (j0+1), col] = -1
        V[(j0+1)*p + k0, col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        W[k0*n + i0, col] = 1
        col += 1

    def emit_schoolbook_2x2x1(i0, j0, k0):
        nonlocal col
        U[i0*m + j0, col] = 1
        V[j0*p + k0, col] = 1
        W[k0*n + i0, col] = 1
        col += 1
        U[i0*m + (j0+1), col] = 1
        V[(j0+1)*p + k0, col] = 1
        W[k0*n + i0, col] = 1
        col += 1
        U[(i0+1)*m + j0, col] = 1
        V[j0*p + k0, col] = 1
        W[k0*n + (i0+1), col] = 1
        col += 1
        U[(i0+1)*m + (j0+1), col] = 1
        V[(j0+1)*p + k0, col] = 1
        W[k0*n + (i0+1), col] = 1
        col += 1

    # 6 Strassen blocks + 2 schoolbook 2x2x1 blocks
    emit_strassen(0, 0, 0)
    emit_strassen(0, 0, 2)
    emit_strassen(0, 0, 4)
    emit_schoolbook_2x2x1(0, 0, 6)
    
    emit_strassen(0, 2, 0)
    emit_strassen(0, 2, 2)
    emit_strassen(0, 2, 4)
    emit_schoolbook_2x2x1(0, 2, 6)
    
    # Verify the initial decomposition
    def verify(U_mat, V_mat, W_mat, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1 if j == k else 0
                    actual = 0
                    for r in range(rank):
                        actual += U_mat[i, r] * V_mat[j, r] * W_mat[k, r]
                    if actual != expected:
                        return False
        return True
    
    if not verify(U, V, W, rank_init):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build the flattened tensor matrix
    # Shape: (n*m*p, rank_init) where each column is a rank-1 tensor
    tensor_mat = np.zeros((n * m * p, rank_init), dtype=np.int64)
    for r in range(rank_init):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_mat[i * m * p + j * p + k, r] = U[i, r] * V[j, r] * W[k, r]
    
    # Find null space over Q using fraction-free Gaussian elimination
    rows = n * m * p
    cols = rank_init
    M = [[Fraction(int(tensor_mat[rr, cc])) for cc in range(cols)] for rr in range(rows)]
    
    # Gaussian elimination to find rank and null space
    rank = 0
    pivot_cols = []
    for c in range(cols):
        if time.time() > DEADLINE - 40:
            break
        pivot = -1
        for r in range(rank, rows):
            if M[r][c] != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        if pivot != rank:
            M[rank], M[pivot] = M[pivot], M[rank]
        for r in range(rank + 1, rows):
            if M[r][c] != 0:
                factor = M[r][c] / M[rank][c]
                for cc in range(c, cols):
                    M[r][cc] -= factor * M[rank][cc]
        pivot_cols.append(c)
        rank += 1
    
    free_cols = [c for c in range(cols) if c not in pivot_cols]
    null_vectors = []
    for fc in free_cols:
        if time.time() > DEADLINE - 35:
            break
        vec = [Fraction(0)] * cols
        vec[fc] = Fraction(1)
        for i in range(rank - 1, -1, -1):
            pc = pivot_cols[i]
            val = M[i][pc]
            s = Fraction(0)
            for cc in range(pc + 1, cols):
                if M[i][cc] != 0 and vec[cc] != 0:
                    s += M[i][cc] * vec[cc]
            vec[pc] = -s / val
        null_vectors.append(vec)
    
    # For each null vector, we can eliminate the free column by adjusting others
    # If sum_r c_r * T_r = 0, then T_fc = -sum_{r != fc} c_r * T_r
    # We can remove term fc by adding c_r * (U_r, V_r, W_r) to remaining terms
    
    best_rank = rank_init
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Try eliminating each free column one at a time
    # For each null vector, the free column can be eliminated
    eliminated = [False] * rank_init
    
    for ns_vec in null_vectors:
        if time.time() > DEADLINE - 25:
            break
        
        # Find the free column in this null vector
        fc = -1
        for c in free_cols:
            if ns_vec[c] != 0:
                fc = c
                break
        if fc == -1 or eliminated[fc]:
            continue
        
        # Check that the coefficients are integers (or at least rational)
        # The null vector gives: sum_r ns_vec[r] * T_r = 0
        # So T_fc = -sum_{r != fc} (ns_vec[r] / ns_vec[fc]) * T_r
        
        # We need to verify that after eliminating, the decomposition still works
        # Create a copy and try to eliminate
        U_test = best_U.copy()
        V_test = best_V.copy()
        W_test = best_W.copy()
        
        # For each other term r, add (ns_vec[r] / ns_vec[fc]) * term_r to term_fc
        # But since we're eliminating term_fc, we add to all other terms
        # Actually: if T_fc + sum_{r!=fc} a_r * T_r = 0 where a_r = ns_vec[r]/ns_vec[fc]
        # Then we can remove term_fc and add a_r * (U_r, V_r, W_r) to... 
        # Wait, this doesn't work directly because the tensor is a sum of products.
        
        # The correct approach: the flattened tensor columns are linearly dependent.
        # If column fc = -sum_{r != fc} c_r * column_r, then we can't simply remove it
        # because the decomposition requires the product structure.
        
        # Instead, we need to find if there's a way to re-express the decomposition
        # with fewer terms. This is more complex.
        
        # Alternative: try to find a subset of columns that spans the same space
        # and verify that the product structure is preserved.
        
        pass
    
    # Simpler approach: try removing each term and checking if the remaining terms
    # can still produce the correct tensor (by adjusting coefficients)
    
    # Actually, the simplest correct approach: if the tensor columns are linearly
    # dependent, we can sometimes find a lower-rank decomposition by solving for
    # new U, V, W matrices.
    
    # For now, let's try a greedy approach: remove terms one by one if possible
    # A term r can be removed if its contribution is zero (all entries are 0)
    
    removable = []
    for r in range(rank_init):
        if np.all(U[:, r] == 0) or np.all(V[:, r] == 0) or np.all(W[:, r] == 0):
            removable.append(r)
    
    if removable:
        keep = [i for i in range(rank_init) if i not in removable]
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        
        if verify(U_new, V_new, W_new, len(keep)):
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    # Fallback: return the rank-50 decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
