import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    deadline = start_time + 110

    def compute_error(U, V, W, r):
        """Compute max absolute error of tensor decomposition."""
        total_tensor = np.zeros((n, m, m, p))
        for c in range(r):
            u_2d = U[:, c].reshape(n, m)
            v_2d = V[:, c].reshape(m, p)
            w_2d = W[:, c].reshape(p, n)
            u_exp = u_2d[:, :, np.newaxis, np.newaxis]
            v_exp = v_2d[np.newaxis, np.newaxis, :, :]
            w_exp = w_2d.T[:, np.newaxis, np.newaxis, :]
            total_tensor += u_exp * v_exp * w_exp
        target = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i, j, j, k] = 1.0
        return np.max(np.abs(total_tensor - target))

    def strassen_2x2_2x5(A_rows, B_rows, U, V, W, offset):
        """
        Add Strassen 2x2*2x2 terms for A (2x2) * B (2x2) part,
        and schoolbook for A (2x2) * B (2x3) part.
        A_rows: tuple of 2 row indices into A (2x4)
        B_rows: tuple of 2 row indices into B (4x5)
        B_cols_first2: first 2 columns of B's block
        B_cols_rest3: remaining 3 columns
        """
        # Strassen for 2x2 * 2x2
        # A = [a00 a01; a10 a11], B = [b00 b01; b10 b11]
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b10)
        # M4 = a11(b10-b01)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        a00_idx = A_rows[0] * m + 0  # row 0, col 0 of A block
        a01_idx = A_rows[0] * m + 1  # row 0, col 1
        a10_idx = A_rows[1] * m + 0  # row 1, col 0
        a11_idx = A_rows[1] * m + 1  # row 1, col 1
        
        # B block: rows B_rows[0], B_rows[1], cols 0,1 (first 2 cols)
        b00_idx = B_rows[0] * p + 0
        b01_idx = B_rows[0] * p + 1
        b10_idx = B_rows[1] * p + 0
        b11_idx = B_rows[1] * p + 1
        
        # For Strassen, we need to express the products in terms of U, V, W
        # Each term in Strassen is a product of a linear combination of A entries
        # and a linear combination of B entries.
        
        # The key: for each Strassen term Mc, we have:
        # Mc = (sum of A entries with coeffs) * (sum of B entries with coeffs)
        # In the tensor decomposition, this becomes:
        # For each pair (i, j) in A and (j', k) in B:
        # U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c]
        
        # Actually, the Strassen algorithm works by replacing 8 multiplications with 7.
        # Each "multiplication" in the original algorithm corresponds to a product of
        # two linear combinations. In the tensor decomposition framework, each such
        # product becomes a rank-1 term (one column in U, V, W).
        
        # Let me think about this more carefully.
        # The 2x2*2x2 multiplication tensor has rank 7 (Strassen).
        # This means we can write:
        # sum_c U_c[i,j] * V_c[j',k] * W_c[k,i] = delta_{j,j'} for the 2x2 case
        # where U_c, V_c, W_c are 2x2 matrices (flattened to 4-dim vectors).
        
        # For the 2x2*2x5 case, we split B into B_left (2x2) and B_right (2x3).
        # A*B = A*B_left + A*B_right
        # A*B_left: rank 7 (Strassen)
        # A*B_right: rank 6 (schoolbook: 2*2*3 = 12? No, 2*3 = 6 scalar products)
        
        # Wait, for A (2x2) * B_right (2x3):
        # C_right[i,k] = sum_j A[i,j] * B_right[j,k]
        # This requires 2*2*3 = 12 scalar multiplications in schoolbook.
        # But we can do better? Actually no, each C[i,k] needs 2 multiplications,
        # and there are 2*3 = 6 entries, so 12 scalar mults.
        # But in the tensor decomposition, each scalar multiplication A[i,j]*B[j,k]
        # contributes to C[i,k] via U[i,j]*V[j,k]*W[k,i].
        # So for A*B_right, we need 2*2*3 = 12 rank-1 terms? No...
        
        # Actually, the tensor for 2x2*2x3 multiplication has rank 2*2*3 = 12
        # in schoolbook. But can we do better? The rank of the 2x2*2x3 tensor
        # is at most 2*rank(2x2*2x1) = 2*4 = 8? Or rank(2x2*2x3) <= rank(2x2*2x2) + rank(2x2*2x1) = 7 + 4 = 11?
        
        # Hmm, let me just use schoolbook for the 2x2*2x3 part: 12 terms.
        # Total per 2x2*2x5: 7 + 12 = 19 terms.
        # Two such blocks: 2*19 = 38 terms. That's worse than 40? No, 38 < 40.
        
        # Wait, I think I'm overcomplicating this. Let me just construct the
        # Strassen-based decomposition explicitly.
        
        # For the 2x2*2x2 Strassen decomposition, the 7 terms are:
        # Term 1: (A00+A11)(B00+B11) -> contributes to C00 and C11
        # Term 2: (A10+A11)B00 -> contributes to C10
        # Term 3: A00(B01-B10) -> contributes to C01
        # Term 4: A11(B10-B01) -> contributes to C00 and C11
        # Term 5: (A00+A01)B11 -> contributes to C00 and C01
        # Term 6: (A10-A00)(B00+B01) -> contributes to C11
        # Term 7: (A01-A11)(B10+B11) -> contributes to C00
        
        # Each term is a product of a linear combo of A entries and a linear combo
        # of B entries. In the tensor decomposition, this means:
        # For term t, U_t[i,j] = coefficient of A[i,j] in the A-linear-combo
        # V_t[j',k] = coefficient of B[j',k] in the B-linear-combo
        # W_t[k,i] = contribution to C[i,k] from this term
        
        # Let me define the 7 Strassen terms for 2x2*2x2:
        # Using 0-indexed: A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        
        strassen_terms = [
            # (U_coeffs for A, V_coeffs for B, W_coeffs for C)
            # Term 1: (a00+a11)(b00+b11), contributes +1 to C00, +1 to C11
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # Term 2: (a10+a11)b00, contributes +1 to C10
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
            # Term 3: a00(b01-b10), contributes +1 to C01
            ({(0,0):1}, {(0,1):1, (1,0):-1}, {(0,1):1}),
            # Term 4: a11(b10-b01), contributes -1 to C00, +1 to C11
            ({(1,1):1}, {(1,0):1, (0,1):-1}, {(0,0):-1, (1,1):1}),
            # Term 5: (a00+a01)b11, contributes -1 to C00, +1 to C01
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # Term 6: (a10-a00)(b00+b01), contributes +1 to C11
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            # Term 7: (a01-a11)(b10+b11), contributes +1 to C00
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        col = offset
        for u_dict, v_dict, w_dict in strassen_terms:
            for (ai, aj), uc in u_dict.items():
                U[a_rows[ai] * m + aj, col] += uc
            for (bj, bk), vc in v_dict.items():
                V[b_rows[bj] * p + bk, col] += vc
            for (ci, ck), wc in w_dict.items():
                W[ck * n + ci, col] += wc
            col += 1
        
        # Schoolbook for A (2x2) * B_right (2x3): 12 terms
        # B_right has columns 2,3,4 (0-indexed) of the 5-column B block
        for ai in range(2):
            for aj in range(2):
                for bk in range(3):
                    bcol = 2 + bk  # columns 2,3,4
                    U[a_rows[ai] * m + aj, col] = 1
                    V[b_rows[aj] * p + bcol, col] = 1
                    W[bcol * n + ai, col] = 1
                    col += 1
        
        return col

    # Build the rank-38 decomposition: 2 blocks of (Strassen 7 + schoolbook 12) = 38
    # Actually, let me reconsider. The 2x4*4x5 multiplication:
    # A is 2x4, B is 4x5, C is 2x5.
    # Split A into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3).
    # Split B into B_top (2x5, rows 0-1) and B_bottom (2x5, rows 2-3).
    # C = A_left * B_top + A_right * B_bottom
    
    # For A_left * B_top (2x2 * 2x5):
    #   Split B_top into B_top_left (2x2, cols 0-1) and B_top_right (2x3, cols 2-4)
    #   A_left * B_top_left: 7 terms (Strassen)
    #   A_left * B_top_right: 12 terms (schoolbook)
    #   Total: 19 terms
    
    # For A_right * B_bottom (2x2 * 2x5):
    #   Split B_bottom into B_bottom_left (2x2, cols 0-1) and B_bottom_right (2x3, cols 2-4)
    #   A_right * B_bottom_left: 7 terms (Strassen)
    #   A_right * B_bottom_right: 12 terms (schoolbook)
    #   Total: 19 terms
    
    # Grand total: 38 terms
    
    rank_38 = 38
    U_38 = np.zeros((n * m, rank_38))
    V_38 = np.zeros((m * p, rank_38))
    W_38 = np.zeros((p * n, rank_38))
    
    # Block 1: A_left (rows 0-1 of A, which is all of A since n=2) * B_top (rows 0-1 of B)
    # A_left uses columns 0-1 of A, so A indices: row 0 -> col 0,1; row 1 -> col 0,1
    # In A (2x4), A_left is A[:, 0:2], so:
    #   A[0,0] -> index 0*4+0 = 0
    #   A[0,1] -> index 0*4+1 = 1
    #   A[1,0] -> index 1*4+0 = 4
    #   A[1,1] -> index 1*4+1 = 5
    # B_top uses rows 0-1 of B, so:
    #   B[0,0] -> 0*5+0 = 0
    #   B[0,1] -> 0*5+1 = 1
    #   B[0,2] -> 0*5+2 = 2
    #   B[0,3] -> 0*5+3 = 3
    #   B[0,4] -> 0*5+4 = 4
    #   B[1,0] -> 1*5+0 = 5
    #   B[1,1] -> 1*5+1 = 6
    #   B[1,2] -> 1*5+2 = 7
    #   B[1,3] -> 1*5+3 = 8
    #   B[1,4] -> 1*5+4 = 9
    
    # For Strassen on A_left * B_top_left:
    # A_left = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # B_top_left = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
    # C contribution to C[0,0], C[0,1], C[1,0], C[1,1]
    
    # I need to map the Strassen terms to the actual indices.
    # Let me write a helper that adds Strassen terms for a specific 2x2*2x2 subproblem.
    
    def add_strassen(A_row0, A_row1, B_row0, B_row1, U, V, W, col_start):
        """
        Add 7 Strassen terms for multiplying A[A_row0:A_row1+1, 0:2] * B[B_row0:B_row1+1, 0:2]
        where A is 2x4 (so columns 0,1 are used) and B is 4x5 (rows B_row0,B_row1, cols 0,1 used).
        A_row0, A_row1: row indices in A (0 or 1)
        B_row0, B_row1: row indices in B (0-3)
        """
        # A entries: A[A_row0, 0], A[A_row0, 1], A[A_row1, 0], A[A_row1, 1]
        # In U: index = A_row * m + col = A_row * 4 + col
        a00 = A_row0 * m + 0
        a01 = A_row0 * m + 1
        a10 = A_row1 * m + 0
        a11 = A_row1 * m + 1
        
        # B entries: B[B_row0, 0], B[B_row0, 1], B[B_row1, 0], B[B_row1, 1]
        # In V: index = B_row * p + col = B_row * 5 + col
        b00 = B_row0 * p + 0
        b01 = B_row0 * p + 1
        b10 = B_row1 * p + 0
        b11 = B_row1 * p + 1
        
        # C entries: C[A_row, B_col] where B_col is 0 or 1
        # In W: index = B_col * n + A_row = B_col * 2 + A_row
        c00 = 0 * n + A_row0  # C[A_row0, 0]
        c01 = 1 * n + A_row0  # C[A_row0, 1]
        c10 = 0 * n + A_row1  # C[A_row1, 0]
        c11 = 1 * n + A_row1  # C[A_row1, 1]
        
        col = col_start
        
        # Term 1: (a00+a11)(b00+b11) -> C00 += 1, C11 += 1
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        # Term 2: (a10+a11)b00 -> C10 += 1
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # Term 3: a00(b01-b10) -> C01 += 1
        U[a00, col] += 1
        V[b01, col] += 1
        V[b10, col] -= 1
        W[c01, col] += 1
        col += 1
        
        # Term 4: a11(b10-b01) -> C00 -= 1, C11 += 1
        U[a11, col] += 1
        V[b10, col] += 1
        V[b01, col] -= 1
        W[c00, col] -= 1
        W[c11, col] += 1
        col += 1
        
        # Term 5: (a00+a01)b11 -> C00 -= 1, C01 += 1
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1
        W[c01, col] += 1
        col += 1
        
        # Term 6: (a10-a00)(b00+b01) -> C11 += 1
        U[a10, col] += 1
        U[a00, col] -= 1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # Term 7: (a01-a11)(b10+b11) -> C00 += 1
        U[a01, col] += 1
        U[a11, col] -= 1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1
        
        return col
    
    def add_schoolbook_2x2_2x3(A_row0, A_row1, B_row0, B_row1, B_col_start, U, V, W, col_start):
        """
        Add schoolbook terms for A[A_row0:A_row1+1, 0:2] * B[B_row0:B_row1+1, B_col_start:B_col_start+3]
        """
        col = col_start
        for ai in range(2):
            arow = [A_row0, A_row1][ai]
            for aj in range(2):
                for bk in range(3):
                    brow = [B_row0, B_row1][aj]
                    bcol = B_col_start + bk
                    U[arow * m + aj, col] = 1
                    V[brow * p + bcol, col] = 1
                    W[bcol * n + arow, col] = 1
                    col += 1
        return col
    
    # Block 1: A_left (A rows 0,1, cols 0,1) * B_top (B rows 0,1)
    col = 0
    # Strassen for A_left * B_top_left (cols 0,1 of B)
    col = add_strassen(0, 1, 0, 1, U_38, V_38, W_38, col)
    # Schoolbook for A_left * B_top_right (cols 2,3,4 of B)
    col = add_schoolbook_2x2_2x3(0, 1, 0, 1, 2, U_38, V_38, W_38, col)
    
    # Block 2: A_right (A rows 0,1, cols 2,3) * B_bottom (B rows 2,3)
    # A_right uses columns 2,3 of A
    # B_bottom uses rows 2,3 of B
    
    # For Strassen, I need to handle A_right which uses cols 2,3 instead of 0,1.
    # Let me generalize add_strassen to accept column offsets.
    
    def add_strassen_general(A_row0, A_row1, A_col_offset, B_row0, B_row1, B_col_offset, U, V, W, col_start):
        a00 = A_row0 * m + (A_col_offset + 0)
        a01 = A_row0 * m + (A_col_offset + 1)
        a10 = A_row1 * m + (A_col_offset + 0)
        a11 = A_row1 * m + (A_col_offset + 1)
        
        b00 = B_row0 * p + (B_col_offset + 0)
        b01 = B_row0 * p + (B_col_offset + 1)
        b10 = B_row1 * p + (B_col_offset + 0)
        b11 = B_row1 * p + (B_col_offset + 1)
        
        c00 = B_col_offset * n + A_row0
        c01 = (B_col_offset + 1) * n + A_row0
        c10 = B_col_offset * n + A_row1
        c11 = (B_col_offset + 1) * n + A_row1
        
        col = col_start
        
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        col += 1
        
        U[a00, col] += 1
        V[b01, col] += 1; V[b10, col] -= 1
        W[c01, col] += 1
        col += 1
        
        U[a11, col] += 1
        V[b10, col] += 1; V[b01, col] -= 1
        W[c00, col] -= 1; W[c11, col] += 1
        col += 1
        
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1
        col += 1
        
        return col
    
    def add_schoolbook_general(A_row0, A_row1, A_col_offset, B_row0, B_row1, B_col_offset, B_col_count, U, V, W, col_start):
        col = col_start
        for ai in range(2):
            arow = [A_row0, A_row1][ai]
            for aj in range(2):
                for bk in range(B_col_count):
                    brow = [B_row0, B_row1][aj]
                    bcol = B_col_offset + bk
                    U[arow * m + (A_col_offset + aj), col] = 1
                    V[brow * p + bcol, col] = 1
                    W[bcol * n + arow, col] = 1
                    col += 1
        return col
    
    # Redo with generalized functions
    U_38 = np.zeros((n * m, rank_38))
    V_38 = np.zeros((m * p, rank_38))
    W_38 = np.zeros((p * n, rank_38))
    
    col = 0
    # Block 1: A[:,0:2] * B[0:2,:]
    col = add_strassen_general(0, 1, 0, 0, 1, 0, U_38, V_38, W_38, col)
    col = add_schoolbook_general(0, 1, 0, 0, 1, 2, 3, U_38, V_38, W_38, col)
    
    # Block 2: A[:,2:4] * B[2:4,:]
    col = add_strassen_general(0, 1, 2, 2, 3, 0, U_38, V_38, W_38, col)
    col = add_schoolbook_general(0, 1, 2, 2, 3, 2, 3, U_38, V_38, W_38, col)
    
    assert col == rank_38, f"Expected {rank_38}, got {col}"
    
    err_38 = compute_error(U_38, V_38, W_38, rank_38)
    if err_38 < 1e-6:
        return {"U": U_38.tolist(), "V": V_38.tolist(), "W": W_38.tolist()}
    
    # Fallback: schoolbook rank 40
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, column] = 1
                V_school[j * p + k, column] = 1
                W_school[k * n + i, column] = 1
                column += 1
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
