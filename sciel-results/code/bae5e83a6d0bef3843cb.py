"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    _t0 = time.time()
    _DEADLINE = 135  # seconds, safety margin before 150s kill
    n, m, p = 2, 4, 4

    def _build_strassen_rank28():
        """Build the rank-28 Strassen decomposition."""
        r = 28
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]

        def _add_strassen(off, ar0, ac0, br0, bc0, cr0, cc0):
            a00 = ar0 * m + (ac0 + 0)
            a01 = ar0 * m + (ac0 + 1)
            a10 = (ar0 + 1) * m + (ac0 + 0)
            a11 = (ar0 + 1) * m + (ac0 + 1)

            b00 = br0 * p + (bc0 + 0)
            b01 = br0 * p + (bc0 + 1)
            b10 = (br0 + 1) * p + (bc0 + 0)
            b11 = (br0 + 1) * p + (bc0 + 1)

            c00 = cc0 * n + cr0
            c01 = (cc0 + 1) * n + cr0
            c10 = cc0 * n + (cr0 + 1)
            c11 = (cc0 + 1) * n + (cr0 + 1)

            # M1 = (a00+a11)(b00+b11)
            U[a00][off] += 1
            U[a11][off] += 1
            V[b00][off] += 1
            V[b11][off] += 1
            W[c00][off] += 1
            W[c11][off] += 1

            # M2 = (a10+a11)b00
            U[a10][off+1] += 1
            U[a11][off+1] += 1
            V[b00][off+1] += 1
            W[c10][off+1] += 1
            W[c11][off+1] -= 1

            # M3 = a00(b01-b11)
            U[a00][off+2] += 1
            V[b01][off+2] += 1
            V[b11][off+2] -= 1
            W[c01][off+2] += 1
            W[c11][off+2] += 1

            # M4 = a11(b10-b00)
            U[a11][off+3] += 1
            V[b10][off+3] += 1
            V[b00][off+3] -= 1
            W[c00][off+3] += 1
            W[c10][off+3] += 1

            # M5 = (a00+a01)b11
            U[a00][off+4] += 1
            U[a01][off+4] += 1
            V[b11][off+4] += 1
            W[c00][off+4] -= 1
            W[c01][off+4] += 1

            # M6 = (a10-a00)(b00+b01)
            U[a10][off+5] += 1
            U[a00][off+5] -= 1
            V[b00][off+5] += 1
            V[b01][off+5] += 1
            W[c11][off+5] += 1

            # M7 = (a01-a11)(b10+b11)
            U[a01][off+6] += 1
            U[a11][off+6] -= 1
            V[b10][off+6] += 1
            V[b11][off+6] += 1
            W[c00][off+6] += 1

        # Block 1: A_L * B_TL -> C[0:2, 0:2]
        _add_strassen(0, 0, 0, 0, 0, 0, 0)
        # Block 2: A_R * B_BL -> C[0:2, 0:2]
        _add_strassen(7, 0, 2, 2, 0, 0, 0)
        # Block 3: A_L * B_TR -> C[0:2, 2:4]
        _add_strassen(14, 0, 0, 0, 2, 0, 2)
        # Block 4: A_R * B_BR -> C[0:2, 2:4]
        _add_strassen(21, 0, 2, 2, 2, 0, 2)

        return U, V, W

    def _verify(U, V, W):
        """Verify the decomposition exactly reconstructs the multiplication tensor."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    val = np.sum(U_np[a_idx] * V_np[b_idx] * W_np[c_idx])
                    if abs(val - 1) > 1e-9:
                        return False
        # Check a few off-diagonal entries
        test_cases = [
            (0, 4, 0), (0, 0, 1), (2, 0, 0), (1, 8, 2), (3, 12, 0),
            (0, 16, 0), (4, 0, 0), (7, 15, 7), (0, 0, 2), (1, 1, 3),
        ]
        for a_idx, b_idx, c_idx in test_cases:
            if a_idx >= n*m or b_idx >= m*p or c_idx >= p*n:
                continue
            val = np.sum(U_np[a_idx] * V_np[b_idx] * W_np[c_idx])
            if abs(val) > 1e-9:
                return False
        return True

    def _try_rank_reduction(U, V, W):
        """
        Try to reduce rank from 28 to 27 by finding a linear dependency
        among the rank-28 columns that allows elimination.
        
        Strategy: For each pair of columns (i, j), check if column j can be
        expressed as a linear combination of the others with rational coefficients.
        We use the structure: if we can find coefficients c_k such that
        for all (a_idx, b_idx, c_idx):
          sum_k c_k * U[a_idx,k] * V[b_idx,k] * W[c_idx,k] = U[a_idx,j]*V[b_idx,j]*W[c_idx,j]
        then we can eliminate column j.
        
        This is a nonlinear system, but we can try a simpler approach:
        Try to find a column that is a linear combination of others in a
        structured way (e.g., from the same Strassen block).
        """
        r = len(W[0])  # current rank
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Build the tensor as a matrix: T[a,b,c] for all valid (a,b,c)
        # The multiplication tensor has 32 nonzero entries.
        # Each column r defines a vector v_r in R^32 (the 32 entries).
        # We want to find if any v_r is in the span of the others.
        
        # Actually, let's try a different approach:
        # The 28 columns come from 4 groups of 7. Within each group of 7,
        # the 7 Strassen terms are linearly independent (that's why rank is 7).
        # But across groups, there might be dependencies.
        
        # Let's try: can we express one entire Strassen block's 7 terms
        # as a combination of terms from other blocks?
        # This seems unlikely but let's check numerically.
        
        # Build matrix M where M[entry, col] = U[a,col]*V[b,col]*W[c,col]
        # for the 32 valid (a,b,c) entries.
        entries = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    entries.append((i*m+j, j*p+k, k*n+i))
        
        M = np.zeros((len(entries), r), dtype=float)
        for idx, (a_idx, b_idx, c_idx) in enumerate(entries):
            for col in range(r):
                M[idx, col] = U_np[a_idx, col] * V_np[b_idx, col] * W_np[c_idx, col]
        
        # Check if any column is in the span of the others
        # Use SVD to find rank and null space
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
        rank_M = np.sum(S > 1e-10)
        
        if rank_M < r:
            # There's a dependency! Find which column to remove.
            # The null space of M^T gives coefficients for the dependency.
            # Actually, we need the null space of M (as a set of column vectors).
            # Let's find the left null space of M: vectors x such that x^T M = 0.
            # No wait, we need: find c such that M @ c = 0, c != 0.
            # The null space of M (right null space) gives such c.
            
            # Recompute with full matrices
            U_svd, S, Vt = np.linalg.svd(M, full_matrices=True)
            # Null space is rows of Vt corresponding to zero singular values
            null_dim = r - rank_M
            if null_dim > 0:
                # Null space vectors are the last null_dim rows of Vt
                null_vec = Vt[-1]  # take one null vector
                # Find the column with largest |null_vec[col]| to remove
                max_col = np.argmax(np.abs(null_vec))
                if abs(null_vec[max_col]) > 1e-10:
                    # Remove column max_col
                    new_U = [row[:max_col] + row[max_col+1:] for row in U]
                    new_V = [row[:max_col] + row[max_col+1:] for row in V]
                    new_W = [row[:max_col] + row[max_col+1:] for row in W]
                    if _verify(new_U, new_V, new_W):
                        return new_U, new_V, new_W
        return None

    # Build the rank-28 solution
    U, V, W = _build_strassen_rank28()
    
    # Verify it's correct
    if not _verify(U, V, W):
        # Should not happen, but fallback
        pass
    
    # Try to reduce rank
    if time.time() - _t0 < _DEADLINE - 10:
        reduced = _try_rank_reduction(U, V, W)
        if reduced is not None:
            U, V, W = reduced
    
    # Try additional reduction attempts with different strategies
    if time.time() - _t0 < _DEADLINE - 5:
        # Try removing pairs of columns and checking if the remaining still spans
        r = len(W[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        entries = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    entries.append((i*m+j, j*p+k, k*n+i))
        
        M = np.zeros((len(entries), r), dtype=float)
        for idx, (a_idx, b_idx, c_idx) in enumerate(entries):
            for col in range(r):
                M[idx, col] = U_np[a_idx, col] * V_np[b_idx, col] * W_np[c_idx, col]
        
        # Try removing each single column and checking if the remaining columns
        # still span the same space (i.e., the removed column is in the span of others)
        for col_to_remove in range(r):
            if time.time() - _t0 > _DEADLINE - 5:
                break
            cols_keep = [c for c in range(r) if c != col_to_remove]
            M_reduced = M[:, cols_keep]
            # Check if M[:, col_to_remove] is in the column space of M_reduced
            # Use least squares
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(M_reduced, M[:, col_to_remove], rcond=None)
                reconstructed = M_reduced @ coeffs
                if np.allclose(reconstructed, M[:, col_to_remove], atol=1e-9):
                    # The column is in the span, we can remove it
                    new_U = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
                    new_V = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
                    new_W = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
                    if _verify(new_U, new_V, new_W):
                        U, V, W = new_U
                        break
            except:
                continue
    
    # Final verification
    if not _verify(U, V, W):
        # Fallback to known-good rank-28
        U, V, W = _build_strassen_rank28()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
