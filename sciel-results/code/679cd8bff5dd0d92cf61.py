import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 84
    # Decomposition:
    # C = A[:, 0:4] B[0:4, :] + A[:, 4] B[4, :]
    # Term 2: M_{3,1,6} -> Rank 18
    # Term 1: M_{3,4,6}
    #   A[:, 0:4] = A_top (2x4) + A_bot (1x4)
    #   M_{1,4,6} -> Rank 24
    #   M_{2,4,6}
    #     B[0:4, :] = B_left (4x2) + B_mid (4x2) + B_right (4x2)
    #     M_{2,4,2} (x3) -> Rank 14 each -> 42
    # Total Rank = 18 + 24 + 42 = 84
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)] # 15 x 84
    V = [[0] * rank for _ in range(m * p)] # 30 x 84
    W = [[0] * rank for _ in range(p * n)] # 18 x 84
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k, val=1):
        nonlocal col
        if col >= rank:
            return
        U[i * m + j][col] = val
        V[j * p + k][col] = val
        W[k * n + i][col] = val
        col += 1

    # 1. M_{3,1,6} part (k=4 in 0-indexed, so j=4)
    # A[:, 4] * B[4, :]
    # i in 0..2, j=4, k in 0..5
    for i in range(3):
        for k in range(6):
            add_term(i, 4, k)
    # 18 terms used. col = 18.

    # 2. M_{1,4,6} part (i=2, j in 0..3, k in 0..5)
    # A[2, 0:4] * B[0:4, :]
    for j in range(4):
        for k in range(6):
            add_term(2, j, k)
    # 24 terms used. col = 18 + 24 = 42.

    # 3. M_{2,4,6} part (i in 0..1, j in 0..3, k in 0..5)
    # Split B into three 4x2 blocks: k in 0..1, 2..3, 4..5
    for block in range(3):
        k_start = block * 2
        k_end = k_start + 2
        # M_{2,4,2} decomposition:
        # A (2x4) = A1 (2x2) + A2 (2x2)
        # B (4x2) = B1 (2x2) + B2 (2x2)
        # C = A1 B1 + A2 B2
        # Use Strassen for each 2x2x2?
        # Strassen terms for 2x2x2:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B21-B11)
        # P4 = A22(B12-B22)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        
        # We need to map these to the global indices.
        # A1 is A[0:2, 0:2], A2 is A[0:2, 2:4]
        # B1 is B[0:2, k_start:k_end], B2 is B[2:4, k_start:k_end]
        
        # This requires linear combinations, so U, V, W entries will be +/- 1.
        
        def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
            # u_coeffs: list of (i, j, coeff)
            # v_coeffs: list of (j, k, coeff)
            # w_coeffs: list of (k, i, coeff)
            nonlocal col
            if col >= rank:
                return
            for (i, j, c) in u_coeffs:
                U[i * m + j][col] += c
            for (j, k, c) in v_coeffs:
                V[j * p + k][col] += c
            for (k, i, c) in w_coeffs:
                W[k * n + i][col] += c
            col += 1

        # Block 1: A1 B1
        # A1: i in 0..1, j in 0..1
        # B1: j in 0..1, k in k_start..k_end
        # Strassen for M_{2,2,2}
        
        # P1: (A00+A11)(B00+B11)
        add_strassen_term(
            [(0, 0, 1), (1, 1, 1)],
            [(0, k_start, 1), (1, k_end, 1)],
            [(k_start, 0, 1), (k_end, 1, 1)]
        )
        # P2: (A10+A11)B00
        add_strassen_term(
            [(1, 0, 1), (1, 1, 1)],
            [(0, k_start, 1)],
            [(k_start, 1, 1)]
        )
        # P3: A00(B10-B00)
        add_strassen_term(
            [(0, 0, 1)],
            [(1, k_start, -1), (0, k_start, 1)], # B10 - B00? No, B10 is B[1,0], B00 is B[0,0]
            [(k_start, 0, 1)]
        )
        # P4: A11(B01-B11)
        add_strassen_term(
            [(1, 1, 1)],
            [(0, k_end, 1), (1, k_end, -1)],
            [(k_end, 1, 1)]
        )
        # P5: (A00+A01)B11
        add_strassen_term(
            [(0, 0, 1), (0, 1, 1)],
            [(1, k_end, 1)],
            [(k_end, 0, 1)]
        )
        # P6: (A10-A00)(B00+B01)
        add_strassen_term(
            [(1, 0, 1), (0, 0, -1)],
            [(0, k_start, 1), (0, k_end, 1)],
            [(k_start, 1, 1), (k_end, 1, 1)]
        )
        # P7: (A01-A11)(B10+B11)
        add_strassen_term(
            [(0, 1, 1), (1, 1, -1)],
            [(1, k_start, 1), (1, k_end, 1)],
            [(k_start, 0, 1), (k_end, 0, 1)]
        )
        
        # Block 2: A2 B2
        # A2: i in 0..1, j in 2..3
        # B2: j in 2..3, k in k_start..k_end
        
        # P1: (A22+A33)(B22+B33) -> Indices: A[0,2], A[1,3] etc.
        add_strassen_term(
            [(0, 2, 1), (1, 3, 1)],
            [(2, k_start, 1), (3, k_end, 1)],
            [(k_start, 0, 1), (k_end, 1, 1)]
        )
        # P2: (A32+A33)B22
        add_strassen_term(
            [(1, 2, 1), (1, 3, 1)],
            [(2, k_start, 1)],
            [(k_start, 1, 1)]
        )
        # P3: A22(B32-B22)
        add_strassen_term(
            [(0, 2, 1)],
            [(3, k_start, -1), (2, k_start, 1)],
            [(k_start, 0, 1)]
        )
        # P4: A33(B23-B33)
        add_strassen_term(
            [(1, 3, 1)],
            [(2, k_end, 1), (3, k_end, -1)],
            [(k_end, 1, 1)]
        )
        # P5: (A22+A23)B33
        add_strassen_term(
            [(0, 2, 1), (0, 3, 1)],
            [(3, k_end, 1)],
            [(k_end, 0, 1)]
        )
        # P6: (A32-A22)(B22+B23)
        add_strassen_term(
            [(1, 2, 1), (0, 2, -1)],
            [(2, k_start, 1), (2, k_end, 1)],
            [(k_start, 1, 1), (k_end, 1, 1)]
        )
        # P7: (A23-A33)(B32+B33)
        add_strassen_term(
            [(0, 3, 1), (1, 3, -1)],
            [(3, k_start, 1), (3, k_end, 1)],
            [(k_start, 0, 1), (k_end, 0, 1)]
        )
        
    return {"U": U, "V": V, "W": W}
