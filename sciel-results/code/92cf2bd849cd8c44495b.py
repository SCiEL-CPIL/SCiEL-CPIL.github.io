"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe (150s wall-clock)
    
    n, m, p = DIMENSIONS
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(U, V, W, i0, j0, k0, col):
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1
        
        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1
        
        # M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        
        return col

    # Helper to add schoolbook terms for a sub-block
    def add_schoolbook(U, V, W, i_range, j_range, k_range, col):
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col

    # Build multiple candidate decompositions and pick the one with lowest rank
    candidates = []
    
    # Candidate 1: Original Strassen+Schoolbook (rank 99)
    rank1 = 99
    U1 = [[0] * rank1 for _ in range(n * m)]
    V1 = [[0] * rank1 for _ in range(m * p)]
    W1 = [[0] * rank1 for _ in range(p * n)]
    
    col = 0
    for k0 in [0, 2, 4]:
        col = add_strassen(U1, V1, W1, 0, 0, k0, col)
    col = add_schoolbook(U1, V1, W1, [0, 1], [0, 1], [6], col)
    col = add_schoolbook(U1, V1, W1, [2], [0, 1], range(7), col)
    
    for k0 in [0, 2, 4]:
        col = add_strassen(U1, V1, W1, 0, 2, k0, col)
    col = add_schoolbook(U1, V1, W1, [0, 1], [2, 3], [6], col)
    col = add_schoolbook(U1, V1, W1, [0, 1], [4], range(7), col)
    col = add_schoolbook(U1, V1, W1, [2], [2, 3, 4], range(7), col)
    
    candidates.append((U1, V1, W1))
    
    # Candidate 2: Use Strassen for all 2x2 blocks where possible
    # Partition j into {0,1}, {2,3}, {4} and k into {0,1}, {2,3}, {4,5}, {6}
    rank2 = 0
    # Count: For each (j_block, k_block) with size 2x2, use Strassen (7 terms)
    # j_blocks: [0,1], [2,3], [4] -> sizes 2,2,1
    # k_blocks: [0,1], [2,3], [4,5], [6] -> sizes 2,2,2,1
    # For i: use Strassen for i in {0,1} (size 2), schoolbook for i=2 (size 1)
    
    # Let's try a different tiling:
    # i_blocks: [0,1], [2]
    # j_blocks: [0,1], [2,3], [4]
    # k_blocks: [0,1], [2,3], [4,5], [6]
    
    # For each combination:
    # i_size x j_size x k_size
    # 2x2x2 -> Strassen (7)
    # 2x2x1 -> 2*2*1=4 schoolbook or... let's just use schoolbook for non-2x2x2
    # 2x1x2 -> 2*1*2=4
    # 2x1x1 -> 2
    # 1x2x2 -> 4
    # 1x2x1 -> 2
    # 1x1x2 -> 2
    # 1x1x1 -> 1
    
    # Total if we use Strassen only for 2x2x2 blocks:
    # i=2: j=2,k=2: 7*4 = 28 (four 2x2x2 blocks)
    # i=2: j=2,k=1: 2*2*1*2 = 8
    # i=2: j=1,k=2: 2*1*2*2 = 8
    # i=2: j=1,k=1: 2*1*1*2 = 4
    # i=1: j=2,k=2: 1*2*2*4 = 16
    # i=1: j=2,k=1: 1*2*1*2 = 4
    # i=1: j=1,k=2: 1*1*2*2 = 4
    # i=1: j=1,k=1: 1*1*1*2 = 2
    # Total: 28+8+8+4+16+4+4+2 = 74
    
    # That's much better! Let's implement this.
    U2 = [[0] * 105 for _ in range(n * m)]  # Start with max possible
    V2 = [[0] * 105 for _ in range(m * p)]
    W2 = [[0] * 105 for _ in range(p * n)]
    
    col = 0
    i_blocks = [(0, 1), (2, 2)]  # (start, end) inclusive
    j_blocks = [(0, 1), (2, 3), (4, 4)]
    k_blocks = [(0, 1), (2, 3), (4, 5), (6, 6)]
    
    for (i0, i1) in i_blocks:
        for (j0, j1) in j_blocks:
            for (k0, k1) in k_blocks:
                i_size = i1 - i0 + 1
                j_size = j1 - j0 + 1
                k_size = k1 - k0 + 1
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    col = add_strassen(U2, V2, W2, i0, j0, k0, col)
                else:
                    for i in range(i0, i1 + 1):
                        for j in range(j0, j1 + 1):
                            for k in range(k0, k1 + 1):
                                U2[i * m + j][col] = 1
                                V2[j * p + k][col] = 1
                                W2[k * n + i][col] = 1
                                col += 1
    
    # Trim to actual size
    U2 = [row[:col] for row in U2]
    V2 = [row[:col] for row in V2]
    W2 = [row[:col] for row in W2]
    candidates.append((U2, V2, W2))
    
    # Candidate 3: Try different tiling - i_blocks [0,1],[2], j_blocks [0,1],[2,3],[4], k_blocks [0,1,2],[3,4,5],[6]
    # 2x2x3, 2x2x3, 2x2x1, 2x2x3, 2x2x3, 2x2x1, 2x1x3, 2x1x3, 2x1x1
    # For 2x2x3: can we use Strassen on the 2x2 part and schoolbook on the 3?
    # Actually let's try: for 2x2xk, use Strassen-like: 7 terms for the 2x2x2 part, then 2*2*(k-2) for the rest
    # Or better: for 2x2x3, use 7 (Strassen for first 2) + 4 (schoolbook for last 1) = 11 vs 12 schoolbook
    # For 2x2x1: 4 schoolbook
    # For 2x1x3: 6 schoolbook
    # For 2x1x1: 2 schoolbook
    
    # Let's try yet another approach: use Strassen on overlapping blocks
    # Actually, let's try the Laderman-like approach for 3x3x3
    
    # For now, let's also try a pure schoolbook baseline as fallback
    U3 = [[0] * (n * m * p) for _ in range(n * m)]
    V3 = [[0] * (n * m * p) for _ in range(m * p)]
    W3 = [[0] * (n * m * p) for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U3[i * m + j][col] = 1
                V3[j * p + k][col] = 1
                W3[k * n + i][col] = 1
                col += 1
    candidates.append((U3, V3, W3))
    
    # Now for each candidate, try to reduce rank via linear algebra
    def build_constraint_matrix_numpy(U, V, W, n, m, p):
        """Build the (n*m*p) x rank matrix using numpy for speed"""
        r = len(U[0])
        M = np.zeros((n * m * p, r), dtype=object)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(r):
                        M[idx, r_idx] = U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                    idx += 1
        return M
    
    def find_redundant_columns_fast(M_np, M_frac, deadline):
        """Use numpy SVD to find rank, then verify with exact arithmetic"""
        rows = M_np.shape[0]
        cols = M_np.shape[1]
        
        # Use float SVD to estimate rank
        M_float = np.array(M_np, dtype=float)
        try:
            sv = np.linalg.svd(M_float, compute_uv=False)
            # Find numerical rank
            tol = max(rows, cols) * np.finfo(float).eps * sv[0] if sv[0] > 0 else 1e-15
            num_rank = np.sum(sv > tol)
        except:
            num_rank = cols
        
        # If SVD says full rank, no reduction possible
        if num_rank >= cols:
            return set(), cols
        
        # Now use exact Gaussian elimination to find which columns are redundant
        # But only check if the rank is actually less than cols
        A = [list(row) for row in M_frac]
        pivot_cols = []
        current_row = 0
        
        for col_idx in range(cols):
            if time.time() > deadline:
                break
            if current_row >= rows:
                continue
            pivot_row = -1
            for row_idx in range(current_row, rows):
                if A[row_idx][col_idx] != 0:
                    pivot_row = row_idx
                    break
            if pivot_row == -1:
                continue
            A[current_row], A[pivot_row] = A[pivot_row], A[current_row]
            pivot_val = A[current_row][col_idx]
            for row_idx in range(current_row + 1, rows):
                if A[row_idx][col_idx] != 0:
                    factor = A[row_idx][col_idx] / pivot_val
                    for c in range(col_idx, cols):
                        A[row_idx][c] -= factor * A[current_row][c]
            pivot_cols.append(col_idx)
            current_row += 1
        
        exact_rank = len(pivot_cols)
        if exact_rank < cols:
            pivot_set = set(pivot_cols)
            redundant = set(c for c in range(cols) if c not in pivot_set)
            return redundant, exact_rank
        else:
            return set(), cols
    
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(len(U[0])):
                        total += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                    if total != 1:
                        return False
        return True
    
    best_result = None
    best_rank = float('inf')
    
    for U_c, V_c, W_c in candidates:
        if time.time() > deadline:
            break
        
        r = len(U_c[0])
        M_frac = build_constraint_matrix_numpy(U_c, V_c, W_c, n, m, p)
        
        redundant, actual_rank = find_redundant_columns_fast(M_frac, M_frac, deadline)
        
        if actual_rank < r and actual_rank < best_rank:
            keep_cols = [c for c in range(r) if c not in redundant]
            U_new = [[U_c[i][c] for c in keep_cols] for i in range(n * m)]
            V_new = [[V_c[i][c] for c in keep_cols] for i in range(m * p)]
            W_new = [[W_c[i][c] for c in keep_cols] for i in range(p * n)]
            
            if verify(U_new, V_new, W_new, n, m, p):
                best_result = {"U": U_new, "V": V_new, "W": W_new}
                best_rank = len(keep_cols)
        
        # Also try without reduction
        if r < best_rank:
            if verify(U_c, V_c, W_c, n, m, p):
                best_result = {"U": U_c, "V": V_c, "W": W_c}
                best_rank = r
    
    if best_result is not None:
        return best_result
    
    # Fallback: return the schoolbook
    return {"U": U3, "V": V3, "W": W3}
# EVOLVE-BLOCK-END
