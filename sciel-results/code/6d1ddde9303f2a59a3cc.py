import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 22 construction
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen parameters for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Coefficients for U (A part), V (B part), W (C part)
    # Local indices i,j in {0,1}, k in {0,1}
    # Global i, j, k
    
    strassen_defs = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: list of (i, j, val)
        # v_coeffs: list of (j, k, val)
        # w_coeffs: list of (k, i, val) -> actually W index is k*n+i
        
        # M1
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (0,1,1), (1,0,1), (1,1,1) ]), # Wait, W is output.
        # C00 += M1, C01 += 0, C10 += 0, C11 += M1? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's define W contributions per M
        # M1: C00 += 1, C11 += 1
        # M2: C10 += 1, C11 -= 1
        # M3: C01 += 1, C11 += 1
        # M4: C00 += 1, C10 += 1
        # M5: C00 -= 1, C01 += 1
        # M6: C11 += 1
        # M7: C00 += 1
    ]
    
    # Correct W coefficients (local k, i)
    # Note: W index is k*n + i. n=2.
    # Local i, k are 0,1.
    
    w_m1 = [(0,0,1), (1,1,1)] # C00, C11
    w_m2 = [(0,1,1), (1,1,-1)] # C10, C11
    w_m3 = [(0,1,1), (1,1,1)] # C01, C11
    w_m4 = [(0,0,1), (0,1,1)] # C00, C10
    w_m5 = [(0,0,-1), (0,1,1)] # C00, C01
    w_m6 = [(1,1,1)] # C11
    w_m7 = [(0,0,1)] # C00
    
    u_m1 = [(0,0,1), (1,1,1)]
    u_m2 = [(1,0,1), (1,1,1)]
    u_m3 = [(0,0,1)]
    u_m4 = [(1,1,1)]
    u_m5 = [(0,0,1), (0,1,1)]
    u_m6 = [(1,0,1), (0,0,-1)]
    u_m7 = [(0,1,1), (1,1,-1)]
    
    v_m1 = [(0,0,1), (1,1,1)]
    v_m2 = [(0,0,1)]
    v_m3 = [(1,0,1), (1,1,-1)]
    v_m4 = [(0,1,1), (0,0,-1)]
    v_m5 = [(1,1,1)]
    v_m6 = [(0,0,1), (0,1,1)]
    v_m7 = [(1,0,1), (1,1,1)]
    
    def add_strassen_block(offset_j, offset_k, start_col):
        # offset_j: which rows of A (0 or 1 in local -> offset in global)
        # offset_k: which cols of B (0 or 1 in local -> offset in global)
        # This block handles A[0:2, offset_j:offset_j+2] * B[offset_j:offset_j+2, offset_k:offset_k+2]
        # Wait, A is 2x3. B is 3x4.
        # Strassen uses A[0:2, 0:2] and B[0:2, 0:2] for first block.
        # So offset_j=0, offset_k=0.
        # Second block: A[0:2, 0:2] and B[0:2, 2:4].
        # So offset_j=0, offset_k=2.
        
        blocks = [
            (u_m1, v_m1, w_m1),
            (u_m2, v_m2, w_m2),
            (u_m3, v_m3, w_m3),
            (u_m4, v_m4, w_m4),
            (u_m5, v_m5, w_m5),
            (u_m6, v_m6, w_m6),
            (u_m7, v_m7, w_m7)
        ]
        
        for idx, (u, v, w) in enumerate(blocks):
            c = start_col + idx
            # Fill U
            for (li, lj, val) in u:
                gi = li
                gj = offset_j + lj
                U[gi * m + gj][c] += val
            # Fill V
            for (lj, lk, val) in v:
                gj = offset_j + lj
                gk = offset_k + lk
                V[gj * p + gk][c] += val
            # Fill W
            for (lk, li, val) in w:
                gk = offset_k + lk
                gi = li
                W[gk * n + gi][c] += val

    # First Strassen block: A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_block(0, 0, 0)
    
    # Second Strassen block: A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_block(0, 2, 7)
    
    # Outer product part: A[0:2, 2:3] * B[2:3, 0:4]
    # A_col = 2. B_row = 2.
    # C[i, k] += A[i, 2] * B[2, k]
    for i in range(2):
        for k in range(4):
            c = 14 + i * 4 + k
            U[i * m + 2][c] = 1
            V[2 * p + k][c] = 1
            W[k * n + i][c] = 1
            
    return {"U": U, "V": V, "W": W}
