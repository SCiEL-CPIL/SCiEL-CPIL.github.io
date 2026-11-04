"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Build the rank-56 Strassen decomposition
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    def build_rank56():
        rank = 56
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        current_rank = 0
        for q in range(4):
            for s in range(7):
                u_loc, v_loc, w_loc = strassen_terms[s]
                r = current_rank
                for i in range(2):
                    for j_loc in range(2):
                        U[i * m + j_loc][r] = u_loc[i * 2 + j_loc]
                for j_loc in range(2):
                    for k_loc in range(2):
                        V[j_loc * p + (2 * q + k_loc)][r] = v_loc[j_loc * 2 + k_loc]
                for i in range(2):
                    for k_loc in range(2):
                        W[(2 * q + k_loc) * n + i][r] = w_loc[k_loc * 2 + i]
                current_rank += 1
        for q in range(4):
            for s in range(7):
                u_loc, v_loc, w_loc = strassen_terms[s]
                r = current_rank
                for i in range(2):
                    for j_loc in range(2):
                        U[i * m + (j_loc + 2)][r] = u_loc[i * 2 + j_loc]
                for j_loc in range(2):
                    for k_loc in range(2):
                        V[(j_loc + 2) * p + (2 * q + k_loc)][r] = v_loc[j_loc * 2 + k_loc]
                for i in range(2):
                    for k_loc in range(2):
                        W[(2 * q + k_loc) * n + i][r] = w_loc[k_loc * 2 + i]
                current_rank += 1
        return U, V, W, rank
    
    def verify_full(U, V, W, rank, tol=1e-9):
        """Fast vectorized verification: sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k].
        Only the 'connected' entries (i_col == j_row) need checking; for those,
        the tensor entry is 1 iff j_col == k_row, else 0."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Flatten index i = i_row*m + i_col, j = j_row*p + j_col, k = k_row*n + k_col
        # Need i_col == j_row (the "inner" index match)
        # For each (i_row, j_row=i_col, j_col, k_row): expected = (j_col == k_row)
        # actual = sum_r U[i_row*m+j_row, r] * V[j_row*p+j_col, r] * W[k_row*n+i_row, r]
        # Vectorize over j_col and k_row for fixed (i_row, j_row):
        #   For fixed i_row, j_row: U_row = U[i_row*m + j_row, :]  (rank,)
        #   V_sub = V[j_row*p:(j_row+1)*p, :]  (p, rank)
        #   W_sub = W[:, i_row]  (p*n, rank) -> reshape to (p, n, rank)
        #   For each j_col, k_row: actual = sum_r U_row[r] * V_sub[j_col, r] * W_sub[k_row, i_row, r]
        #   = sum_r U_row[r] * V_sub[j_col, r] * W_sub[k_row, i_row, r]
        #   Define C[j_col, k_row] = sum_r U_row[r] * V_sub[j_col, r] * W_sub[k_row, i_row, r]
        #   This is: C = (V_sub * U_row[None,:]) @ W_sub[:,:,i_row].T  -- no, let's think differently
        #   C[j_col, k_row] = sum_r (U_row[r] * V_sub[j_col, r]) * W_sub[k_row, i_row, r]
        #   Let A[j_col, r] = V_sub[j_col, r] * U_row[r]  (p, rank)
        #   Let B[k_row, r] = W_sub[k_row, i_row, r]  (p, rank)  [W_sub is (p, n, rank), take [:, i_row, :]]
        #   C[j_col, k_row] = sum_r A[j_col, r] * B[k_row, r] = (A @ B.T)[j_col, k_row]
        #   Expected: C[j_col, k_row] = 1 if j_col == k_row else 0, i.e., C should be identity (p x p)
        for i_row in range(n):
            for j_row in range(m):
                i_idx = i_row * m + j_row
                U_row = U[i_idx, :]  # (rank,)
                V_sub = V[j_row * p:(j_row + 1) * p, :]  # (p, rank)
                W_sub = W.reshape(p, n, rank)[:, i_row, :]  # (p, rank)
                A = V_sub * U_row[None, :]  # (p, rank)
                C = A @ W_sub.T  # (p, p)
                expected = np.eye(p)
                if not np.allclose(C, expected, atol=tol):
                    return False
        return True
    
    # Start with rank-56 solution
    U, V, W, rank = build_rank56()
    
    # Verify the base decomposition
    if not verify_full(U, V, W, rank):
        # Fallback: return as-is
        U_list = [[int(U[i, r]) for r in range(rank)] for i in range(n * m)]
        V_list = [[int(V[i, r]) for r in range(rank)] for i in range(m * p)]
        W_list = [[int(W[i, r]) for r in range(rank)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # The 56 columns are organized as 8 blocks (q=0..3, half=0/1) x 7 Strassen terms (s=0..6).
    # Column index r = q*7 + s for first 4 q's (j<2), then r = 28 + q*7 + s for second 4 q's (j>=2).
    # Actually from build_rank56: first loop q=0..3, s=0..6 -> r=0..27 (j_loc=0,1 i.e. j<2)
    # Second loop q=0..3, s=0..6 -> r=28..55 (j_loc=2,3 i.e. j>=2)
    # So column r: if r < 28: q = r//7, s = r%7, half=0 (j in {0,1})
    #              if r >= 28: q = (r-28)//7, s = (r-28)%7, half=1 (j in {2,3})
    
    def col_to_block(r):
        if r < 28:
            return r // 7, r % 7, 0
        else:
            return (r - 28) // 7, (r - 28) % 7, 1
    
    def try_remove(columns_to_remove):
        """Try removing the given columns; return True if still valid."""
        mask = np.ones(rank, dtype=bool)
        for c in columns_to_remove:
            mask[c] = False
        U_t = U[:, mask]
        V_t = V[:, mask]
        W_t = W[:, mask]
        return verify_full(U_t, V_t, W_t, U_t.shape[1])
    
    # Phase 1: Try removing single columns (fast now with vectorized verify)
    for r_to_remove in range(rank):
        if time.time() > deadline:
            break
        if try_remove([r_to_remove]):
            mask = np.ones(rank, dtype=bool)
            mask[r_to_remove] = False
            U, V, W = U[:, mask], V[:, mask], W[:, mask]
            rank = rank - 1
            best_rank = rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Phase 2: Structured cross-block elimination.
    # The 8 blocks are (q, half) for q in 0..3, half in {0,1}.
    # Blocks with the same Strassen term s but different (q, half) may have redundancy
    # because the U/V/W entries repeat the same Strassen pattern.
    # Try removing pairs of columns that share the same s but different (q, half).
    
    # Rebuild block structure after Phase 1 removals
    # We need to track which columns remain and their block assignments
    # For efficiency, just try all pairs of remaining columns that are "similar"
    
    if time.time() < deadline - 20:
        remaining = list(range(rank))
        # Group by Strassen term s
        for s in range(7):
            cols_with_s = [r for r in remaining if col_to_block(r)[1] == s]
            # Try removing pairs from this group
            for idx1 in range(len(cols_with_s)):
                if time.time() > deadline:
                    break
                for idx2 in range(idx1 + 1, len(cols_with_s)):
                    c1, c2 = cols_with_s[idx1], cols_with_s[idx2]
                    if try_remove([c1, c2]):
                        mask = np.ones(rank, dtype=bool)
                        mask[c1] = False
                        mask[c2] = False
                        U, V, W = U[:, mask], V[:, mask], W[:, mask]
                        rank = rank - 2
                        best_rank = rank
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        remaining = [r for r in range(rank)]
                        break
                if time.time() > deadline:
                    break
    
    # Phase 3: Try removing triples from same-s groups
    if time.time() < deadline - 10:
        remaining = list(range(rank))
        for s in range(7):
            cols_with_s = [r for r in remaining if col_to_block(r)[1] == s]
            if len(cols_with_s) < 3:
                continue
            # Try all triples
            for i1 in range(len(cols_with_s)):
                if time.time() > deadline:
                    break
                for i2 in range(i1 + 1, len(cols_with_s)):
                    if time.time() > deadline:
                        break
                    for i3 in range(i2 + 1, len(cols_with_s)):
                        c1, c2, c3 = cols_with_s[i1], cols_with_s[i2], cols_with_s[i3]
                        if try_remove([c1, c2, c3]):
                            mask = np.ones(rank, dtype=bool)
                            mask[c1] = False
                            mask[c2] = False
                            mask[c3] = False
                            U, V, W = U[:, mask], V[:, mask], W[:, mask]
                            rank = rank - 3
                            best_rank = rank
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            remaining = [r for r in range(rank)]
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
    
    # Convert to lists with proper integer/half-integer handling
    def to_list(mat, rank_val):
        result = []
        for i in range(mat.shape[0]):
            row = []
            for r in range(rank_val):
                val = mat[i, r]
                # Round to nearest 0.5
                val_rounded = round(val * 2) / 2
                # Check if it's an integer or half-integer
                if abs(val_rounded * 2 - round(val_rounded * 2)) < 1e-6:
                    if abs(val_rounded - round(val_rounded)) < 1e-6:
                        row.append(int(round(val_rounded)))
                    else:
                        row.append(round(val_rounded * 2) / 2)
                else:
                    row.append(round(val_rounded * 2) / 2)
            result.append(row)
        return result
    
    U_list = to_list(best_U, best_rank)
    V_list = to_list(best_V, best_rank)
    W_list = to_list(best_W, best_rank)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
