import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)

    # We use a block-Strassen decomposition.
    # Split A (2x4) into A_L (2x2) and A_R (2x2).
    # Split B (4x8) into B_T (2x8) and B_B (2x8).
    # C = A_L * B_T + A_R * B_B.
    # Each term is (2x2) * (2x8).
    # Split (2x8) into four (2x2) blocks.
    # So we have 8 independent (2x2) * (2x2) multiplications.
    # Each uses Strassen with rank 7.
    # Total rank = 8 * 7 = 56.

    rank = 56
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen coefficients for 2x2 * 2x2
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]], C = A*B
    # M1=(a0+a3)(b0+b3), M2=(a2+a3)b0, M3=a0(b1-b3),
    # M4=a3(b2-b0), M5=(a0+a1)b3, M6=(a2-a0)(b0+b1),
    # M7=(a1-a3)(b2+b3)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    
    # Indices in local 2x2 block:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    u_coeffs = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    v_coeffs = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    w_coeffs = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1],
        [1, 0, 1, 0], [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    # 8 blocks:
    # 0-3: A_L (cols 0,1 of A), B_T (rows 0,1 of B)
    #   - Block 0: B cols 0,1
    #   - Block 1: B cols 2,3
    #   - Block 2: B cols 4,5
    #   - Block 3: B cols 6,7
    # 4-7: A_R (cols 2,3 of A), B_B (rows 2,3 of B)
    #   - Block 4: B cols 0,1
    #   - Block 5: B cols 2,3
    #   - Block 6: B cols 4,5
    #   - Block 7: B cols 6,7

    for t in range(8):
        if t < 4:
            a_row_off = 0 # A_L is rows 0,1 of A (which are all rows)
            a_col_off = 0 # A_L is cols 0,1
            b_row_off = 0 # B_T is rows 0,1
        else:
            a_row_off = 0 # A_R is rows 0,1
            a_col_off = 2 # A_R is cols 2,3
            b_row_off = 2 # B_B is rows 2,3
            
        b_col_off = (t % 4) * 2

        # U indices: (n*m, rank) -> (8, rank)
        # A is 2x4. Flattened: [A00, A01, A02, A03, A10, A11, A12, A13]
        # Local 2x2 indices:
        # (0,0) -> a_row_off + 0, a_col_off + 0 -> index 0*4 + 0? No.
        # Index in U for A[i,j] is i*m + j.
        
        u_idx = [
            (0 + a_row_off) * m + (0 + a_col_off),      # a0
            (0 + a_row_off) * m + (1 + a_col_off),      # a1
            (1 + a_row_off) * m + (0 + a_col_off),      # a2
            (1 + a_row_off) * m + (1 + a_col_off)       # a3
        ]
        
        # V indices: (m*p, rank) -> (32, rank)
        # B is 4x8. Flattened: [B00..B07, B10..B17, B20..B27, B30..B37]
        # Index in V for B[i,j] is i*p + j.
        v_idx = [
            (0 + b_row_off) * p + (0 + b_col_off),      # b0
            (0 + b_row_off) * p + (1 + b_col_off),      # b1
            (1 + b_row_off) * p + (0 + b_col_off),      # b2
            (1 + b_row_off) * p + (1 + b_col_off)       # b3
        ]
        
        # W indices: (p*n, rank) -> (16, rank)
        # C is 2x8 (p*n = 8*2 = 16).
        # Convention: W[k*n + i, r] corresponds to C[i, k]?
        # "official trace convention W[k*n+i,r]"
        # Usually C[i,j] is at index i*p + j in row-major.
        # But here W is (p*n, rank).
        # Let's check the seed program's W index usage.
        # c_idx = [b_col_off * n, (b_col_off + 1) * n, b_col_off * n + 1, (b_col_off + 1) * n + 1]
        # b_col_off is the column index in B, which corresponds to the column index in C (since B is 4x8, C is 2x8).
        # So C has columns 0..7.
        # b_col_off is 0, 2, 4, 6.
        # The block covers C columns b_col_off and b_col_off+1.
        # C rows are 0 and 1.
        # The index in W is k*n + i.
        # If k is the column index of C (0..7) and i is the row index of C (0..1).
        # Then index = col * 2 + row.
        # Let's verify:
        # b_col_off = 0.
        # C[0,0] -> 0*2+0=0.
        # C[1,0] -> 0*2+1=1.
        # C[0,1] -> 1*2+0=2.
        # C[1,1] -> 1*2+1=3.
        # The seed program used:
        # c_idx = [0, 2, 1, 3] for b_col_off=0?
        # Seed: c_idx = [b_col_off * n, (b_col_off + 1) * n, b_col_off * n + 1, (b_col_off + 1) * n + 1]
        # For b_col_off=0: [0, 2, 1, 3].
        # This maps:
        # 0 -> C[0,0] (row 0, col 0) if index = col*n + row? 0*2+0=0. Yes.
        # 2 -> C[0,1] (row 0, col 1)? 1*2+0=2. Yes.
        # 1 -> C[1,0] (row 1, col 0)? 0*2+1=1. Yes.
        # 3 -> C[1,1] (row 1, col 1)? 1*2+1=3. Yes.
        # So the order in w_coeffs corresponds to:
        # w_coeffs[0] adds to C[0,0] and C[1,1]?
        # Let's check w_coeffs[0] = [1, 0, 0, 1].
        # It adds to indices 0 and 3.
        # Index 0 is C[0,0]. Index 3 is C[1,1].
        # Strassen M1 contributes to C00 and C11.
        # C00 = M1 + ...
        # C11 = M1 - ...
        # So M1 adds +1 to C00 and +1 to C11.
        # This matches w_coeffs[0] adding to index 0 (C00) and index 3 (C11).
        # So the mapping is correct.

        w_idx = [
            (0 + b_col_off) * n + 0,      # C[0, b_col_off]
            (1 + b_col_off) * n + 0,      # C[0, b_col_off+1]
            (0 + b_col_off) * n + 1,      # C[1, b_col_off]
            (1 + b_col_off) * n + 1       # C[1, b_col_off+1]
        ]
        
        # Wait, the seed program's w_idx order was:
        # [b_col_off * n, (b_col_off + 1) * n, b_col_off * n + 1, (b_col_off + 1) * n + 1]
        # Which is:
        # 0: C[0, b_col_off]
        # 1: C[0, b_col_off+1]
        # 2: C[1, b_col_off]
        # 3: C[1, b_col_off+1]
        
        # My w_idx above:
        # 0: (0+b_col_off)*n + 0 = C[0, b_col_off]
        # 1: (1+b_col_off)*n + 0 = C[0, b_col_off+1]
        # 2: (0+b_col_off)*n + 1 = C[1, b_col_off]
        # 3: (1+b_col_off)*n + 1 = C[1, b_col_off+1]
        
        # This matches the seed program's order.
        # And w_coeffs are defined for C00, C01, C10, C11?
        # Let's re-verify w_coeffs mapping.
        # w_coeffs[0] = [1, 0, 0, 1] -> adds to C00 and C11.
        # In my w_idx:
        # Index 0 is C00.
        # Index 3 is C11.
        # So w_coeffs[0][0] adds to w_idx[0] (C00).
        # w_coeffs[0][3] adds to w_idx[3] (C11).
        # This is correct.
        
        for l in range(7):
            r = t * 7 + l
            for li, c in enumerate(u_coeffs[l]):
                if c:
                    U[u_idx[li], r] += c
            for li, c in enumerate(v_coeffs[l]):
                if c:
                    V[v_idx[li], r] += c
            for li, c in enumerate(w_coeffs[l]):
                if c:
                    W[w_idx[li], r] += c

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
