import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Rank 49 decomposition for 4x4x4 using Kronecker product of Strassen
    # Then duplicate for 4x4x8 to get rank 98
    
    # Strassen 2x2x2 decomposition
    # U, V, W are 4x7
    U2 = [
        [1, 0, 1, 0, 1, 1, 0], # A00
        [0, 0, 0, 0, 0, 0, 0], # A01 - wait, Strassen forms
        [0, 1, 0, 0, 0, 0, 0], # A10
        [0, 0, 0, 1, 0, 0, 1]  # A11
    ]
    # Correct Strassen U (coefficients for A)
    # M1 = (A00+A11)(B00+B11) -> A00, A11
    # M2 = (A10+A11)B00 -> A10, A11
    # M3 = A00(B01-B11) -> A00
    # M4 = A11(B10-B00) -> A11
    # M5 = (A00+A10)B11 -> A00, A10
    # M6 = (A01-A00)(B00+B10) -> A01, A00
    # M7 = (A10-A11)(B11+B01) -> A10, A11
    
    # Rows: A00, A01, A10, A11
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 0, 1, 0],  # A01
        [0, 1, 0, 0, 1, 0, 1],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    # V (coefficients for B)
    # M1: B00, B11
    # M2: B00
    # M3: B01, B11
    # M4: B10, B00
    # M5: B11
    # M6: B00, B10
    # M7: B11, B01
    
    # Rows: B00, B01, B10, B11
    V2 = [
        [1, 0, 0, -1, 0, 1, 0], # B00
        [0, 1, 0, 0, 0, 0, 1],  # B01
        [0, 0, 1, 1, 0, 1, 0],  # B10
        [1, -1, 0, 0, 1, 0, 1]  # B11
    ]
    
    # W (coefficients for C)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Rows: C00, C01, C10, C11
    # Trace convention W[k*n+i, r]. For 2x2, n=2.
    # k=0, i=0 -> C00. k=0, i=1 -> C01? No.
    # C is 2x2. C[k, i]?
    # Standard matrix mult C = AB. C_{ij} = sum A_{ik} B_{kj}.
    # Trace convention W[k*n+i, r].
    # For 2x2, n=2.
    # Row 0: k=0, i=0 -> C_{00}
    # Row 1: k=0, i=1 -> C_{01}
    # Row 2: k=1, i=0 -> C_{10}
    # Row 3: k=1, i=1 -> C_{11}
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Verify U2, V2, W2
    # This is standard Strassen.
    
    # Construct 4x4x4 decomposition
    n4, m4, p4 = 4, 4, 4
    rank4 = 49
    U4 = [[0] * rank4 for _ in range(16)]
    V4 = [[0] * rank4 for _ in range(16)]
    W4 = [[0] * rank4 for _ in range(16)]
    
    for r in range(7):
        for s in range(7):
            c = 7 * r + s
            for i_out in range(2):
                for j_out in range(2):
                    for i_in in range(2):
                        for j_in in range(2):
                            # U4 row index: 4 * (2*i_out + i_in) + (2*j_out + j_in)
                            row_u = 4 * (2 * i_out + i_in) + (2 * j_out + j_in)
                            U4[row_u][c] = U2[2 * i_out + j_out][r] * U2[2 * i_in + j_in][s]
                            
                            # V4 row index: 4 * (2*j_out + j_in) + (2*k_out + k_in)
                            # Wait, V is for B. B indices j, k.
                            # In the loop above, we only have i, j.
                            # We need k.
                            pass
                            
    # The loop above is incomplete. We need to iterate over k as well for V and W.
    # U depends on i, j.
    # V depends on j, k.
    # W depends on k, i.
    
    U4 = [[0] * rank4 for _ in range(16)]
    V4 = [[0] * rank4 for _ in range(16)]
    W4 = [[0] * rank4 for _ in range(16)]
    
    for r in range(7):
        for s in range(7):
            c = 7 * r + s
            for i_out in range(2):
                for j_out in range(2):
                    for k_out in range(2):
                        for i_in in range(2):
                            for j_in in range(2):
                                for k_in in range(2):
                                    # U4
                                    row_u = 4 * (2 * i_out + i_in) + (2 * j_out + j_in)
                                    U4[row_u][c] += U2[2 * i_out + j_out][r] * U2[2 * i_in + j_in][s]
                                    
                                    # V4
                                    row_v = 4 * (2 * j_out + j_in) + (2 * k_out + k_in)
                                    V4[row_v][c] += V2[2 * j_out + k_out][r] * V2[2 * j_in + k_in][s]
                                    
                                    # W4
                                    row_w = 4 * (2 * k_out + k_in) + (2 * i_out + i_in)
                                    W4[row_w][c] += W2[2 * k_out + i_out][r] * W2[2 * k_in + i_in][s]
                                    
    # Now construct 4x4x8 decomposition
    n, m, p = 4, 4, 8
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U is same for both blocks
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
            U[i][c + 49] = U4[i][c]
            
    # V
    # Rows: j * 8 + k
    for j in range(4):
        for k in range(8):
            row_v = j * 8 + k
            if k < 4:
                # First block
                row_v4 = j * 4 + k
                for c in range(49):
                    V[row_v][c] = V4[row_v4][c]
                    V[row_v][c + 49] = 0
            else:
                # Second block
                k2 = k - 4
                row_v4 = j * 4 + k2
                for c in range(49):
                    V[row_v][c] = 0
                    V[row_v][c + 49] = V4[row_v4][c]
                    
    # W
    # Rows: k * 4 + i
    for k in range(8):
        for i in range(4):
            row_w = k * 4 + i
            if k < 4:
                # First block
                row_w4 = k * 4 + i
                for c in range(49):
                    W[row_w][c] = W4[row_w4][c]
                    W[row_w][c + 49] = 0
            else:
                # Second block
                k2 = k - 4
                row_w4 = k2 * 4 + i
                for c in range(49):
                    W[row_w][c] = 0
                    W[row_w][c + 49] = W4[row_w4][c]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
