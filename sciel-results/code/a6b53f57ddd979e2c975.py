import numpy as np
import time

n, m, p = 4, 4, 6

def construct():
    start = time.time()
    DEADLINE = 118.0  # safety margin before 120s

    # A is 4x4, B is 4x6, C is 4x6
    # A_flat index: a * 4 + c
    # B_flat index: c * 6 + d
    # C_flat index: d * 4 + a  (trace convention W[k*n+i])

    def a_idx(r, c):
        return r * m + c

    def b_idx(r, c):
        return r * p + c

    def c_idx(r, c):
        return c * n + r

    # Strassen 2x2 * 2x2 decomposition (rank 7)
    # Each term: (u_dict, v_dict, w_dict)
    # u_dict: {(row, col): coeff} for the 2x2 A-block
    # v_dict: {(row, col): coeff} for the 2x2 B-block
    # w_dict: {(row, col): coeff} for the 2x2 C-block
    strassen_2x2 = [
        # M1 = (A00 + A11)(B00 + B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10 + A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00(B01 - B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11(B10 - B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00 + A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10 - A00)(B00 + B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01 - A11)(B10 + B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    # We'll build a rank-77 decomposition:
    # Part 1: 4x4 * 4x4 (first 4 columns of B) -> rank 49 (Strassen at two levels)
    # Part 2: 4x4 * 4x2 (last 2 columns of B) -> rank 28
    # Total: 77

    # For Part 2, we use 4 independent 2x2 * 2x2 Strassen decompositions:
    # The 4x4 * 4x2 multiplication can be viewed as:
    #   C_top[2x2] = A_top[2x2] * B_top[2x2]   (rows 0-1 of C, cols 4-5)
    #   C_bot[2x2] = A_bot[2x2] * B_bot[2x2]   (rows 2-3 of C, cols 4-5)
    # But wait, that's only 2x2 * 2x2 for each, which is rank 7 each, total 14.
    # That's not enough. We need to handle the full 4x4 * 4x2.
    #
    # Actually, 4x4 * 4x2 means:
    #   C[i, 4] = sum_{j=0}^{3} A[i,j] * B[j,4]  for i=0..3
    #   C[i, 5] = sum_{j=0}^{3} A[i,j] * B[j,5]  for i=0..3
    #
    # This is equivalent to two separate 4x4 * 4x1 multiplications.
    # Each 4x4 * 4x1 has rank 16 (schoolbook).
    # But we can do better.
    #
    # Let's use a different approach: treat 4x4 * 4x2 as 2x2 blocks.
    # A is 4x4, B is 4x2 (last 2 columns).
    # Split A into 2x2 blocks: A00, A01, A10, A11 (each 2x2)
    # B is 4x2, split into 2x2 blocks: B0 (rows 0-1), B1 (rows 2-3)
    # C is 4x2, split into 2x2 blocks: C0 (rows 0-1), C1 (rows 2-3)
    #
    # C0 = A00 * B0 + A01 * B1
    # C1 = A10 * B0 + A11 * B1
    #
    # This is 4 separate 2x2 * 2x2 multiplications, each rank 7, total rank 28.
    # Perfect!

    rank = 77
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0

    # Part 1: 4x4 * 4x4 (columns 0-3 of B) using recursive Strassen (rank 49)
    # Outer Strassen operates on 2x2 blocks of 2x2 submatrices
    # Inner Strassen operates on individual elements within each 2x2 submatrix
    
    for outer_step in strassen_2x2:
        a_coeffs, b_coeffs, c_coeffs = outer_step
        for inner_step in strassen_2x2:
            ia_coeffs, ib_coeffs, ic_coeffs = inner_step

            # Build U contribution
            for (oi, oj), av in a_coeffs.items():
                if av == 0:
                    continue
                for (ii, ij), iav in ia_coeffs.items():
                    if iav == 0:
                        continue
                    # A element at row (2*oi + ii), col (2*oj + ij)
                    r = 2 * oi + ii
                    c = 2 * oj + ij
                    U[a_idx(r, c), col] += av * iav

            # Build V contribution (only first 4 columns of B)
            for (oj, ok), bv in b_coeffs.items():
                if bv == 0:
                    continue
                for (ij, ik), ibv in ib_coeffs.items():
                    if ibv == 0:
                        continue
                    # B element at row (2*oj + ij), col (2*ok + ik)
                    r = 2 * oj + ij
                    c = 2 * ok + ik
                    if c < 4:  # Only first 4 columns
                        V[b_idx(r, c), col] += bv * ibv

            # Build W contribution (only first 4 columns of C)
            for (oi, ok), cv in c_coeffs.items():
                if cv == 0:
                    continue
                for (ii, ik), icv in ic_coeffs.items():
                    if icv == 0:
                        continue
                    # C element at row (2*oi + ii), col (2*ok + ik)
                    r = 2 * oi + ii
                    c = 2 * ok + ik
                    if c < 4:  # Only first 4 columns
                        W[c_idx(r, c), col] += cv * icv

            col += 1

    # Part 2: 4x4 * 4x2 (columns 4-5 of B) using 4 independent 2x2 * 2x2 Strassen (rank 28)
    # C0 (rows 0-1, cols 4-5) = A00 * B0 + A01 * B1
    # C1 (rows 2-3, cols 4-5) = A10 * B0 + A11 * B1
    # where:
    #   A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    #   A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    #   B0 = B[0:2, 4:6], B1 = B[2:4, 4:6]
    #   C0 = C[0:2, 4:6], C1 = C[2:4, 4:6]
    
    # For each of the 4 combinations:
    # (A_block, B_block, C_block) where:
    # A_block is one of A00, A01, A10, A11
    # B_block is one of B0, B1
    # C_block is one of C0, C1
    # The valid combinations are:
    # (A00, B0, C0), (A01, B1, C0), (A10, B0, C1), (A11, B1, C1)
    
    # For each valid combination, apply Strassen 2x2 * 2x2 (7 terms)
    # Total: 4 * 7 = 28 terms
    
    a_blocks = [
        (0, 0),  # A00: rows 0-1, cols 0-1
        (0, 2),  # A01: rows 0-1, cols 2-3
        (2, 0),  # A10: rows 2-3, cols 0-1
        (2, 2),  # A11: rows 2-3, cols 2-3
    ]
    
    b_blocks = [
        (0, 4),  # B0: rows 0-1, cols 4-5
        (2, 4),  # B1: rows 2-3, cols 4-5
    ]
    
    c_blocks = [
        (0, 4),  # C0: rows 0-1, cols 4-5
        (2, 4),  # C1: rows 2-3, cols 4-5
    ]
    
    # Valid combinations: (A_block_idx, B_block_idx, C_block_idx)
    valid_combos = [
        (0, 0, 0),  # A00 * B0 -> C0
        (1, 1, 0),  # A01 * B1 -> C0
        (2, 0, 1),  # A10 * B0 -> C1
        (3, 1, 1),  # A11 * B1 -> C1
    ]
    
    for a_idx_block, b_idx_block, c_idx_block in valid_combos:
        a_row_start, a_col_start = a_blocks[a_idx_block]
        b_row_start, b_col_start = b_blocks[b_idx_block]
        c_row_start, c_col_start = c_blocks[c_idx_block]
        
        for u_pat, v_pat, w_pat in strassen_2x2:
            # Build U contribution
            for (r, c), coeff in u_pat.items():
                i = a_row_start + r
                j = a_col_start + c
                if 0 <= i < n and 0 <= j < m:
                    U[a_idx(i, j), col] += coeff
            
            # Build V contribution
            for (r, c), coeff in v_pat.items():
                i = b_row_start + r
                j = b_col_start + c
                if 0 <= i < m and 0 <= j < p:
                    V[b_idx(i, j), col] += coeff
            
            # Build W contribution
            for (r, c), coeff in w_pat.items():
                i = c_row_start + r
                j = c_col_start + c
                if 0 <= i < n and 0 <= j < p:
                    W[c_idx(i, j), col] += coeff
            
            col += 1

    # Verify the decomposition
    try:
        rng = np.random.default_rng(42)
        for _ in range(3):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            A_flat = A.reshape(n * m)
            B_flat = B.reshape(m * p)
            C_true = A @ B
            C_true_flat = C_true.T.reshape(p * n)
            a = U.T @ A_flat
            b = V.T @ B_flat
            C_scheme_flat = W.T @ (a * b)
            if not np.allclose(C_scheme_flat, C_true_flat, atol=1e-8):
                break
    except Exception:
        pass

    if time.time() - start > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
