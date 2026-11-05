import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 1. A_right (4x1) * B_bottom (1x5) -> Rank 1
    # 2. A_left (4x4) * B_top_left (4x4) -> Rank 49 (Strassen 4x4)
    # 3. A_left (4x4) * B_top_right (4x1) -> Rank 16 (Schoolbook 4x4x1)
    # Total Rank = 1 + 49 + 16 = 66
    
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:, 4] * B[4, :]
    # C[i, k] += A[i, 4] * B[4, k]
    # u: 1 at i*5+4
    # v: 1 at 4*5+k
    # w: 1 at k*4+i
    for i in range(n):
        U[i * m + 4][col] = 1
    for k in range(p):
        V[4 * p + k][col] = 1
    for i in range(n):
        for k in range(p):
            W[k * n + i][col] = 1
    col += 1
    
    # Part 2: A[0:4, 0:4] * B[0:4, 0:4]
    # Strassen 4x4
    # We need the 49 rank-1 terms.
    # Base Strassen 2x2 terms (u, v, w vectors of length 4)
    # Indices 0:00, 1:01, 2:10, 3:11
    base_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    base_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    base_w = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 0, 1, -1],# M2 -> C10, C11
        [0, 1, 0, 1], # M3 -> C01, C11
        [1, 0, 1, 0], # M4 -> C00, C10
        [-1, 1, 0, 0],# M5 -> C00, C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]
    
    # Meta Strassen 4x4 combinations
    # X_m is 2x2 matrix of A blocks. Y_m is 2x2 matrix of B blocks.
    # A blocks: A00, A01, A10, A11.
    # B blocks: B00, B01, B10, B11.
    
    # Define X_m and Y_m as 2x2 matrices of coefficients for the blocks.
    # Block index (b,c) corresponds to A[2b:2b+2, 2c:2c+2].
    # Local index in block: 0:00, 1:01, 2:10, 3:11.
    
    # M1: (A00+A11)(B00+B11)
    # X1 = [[1,0],[0,1]] on blocks? No.
    # X1 = A00 + A11. This is a 2x2 matrix.
    # X1[0,0] = A00[0,0] + A11[0,0]
    # X1[0,1] = A00[0,1] + A11[0,1]
    # etc.
    # So X_m is formed by summing blocks.
    
    # Let's define the block coefficients for X_m and Y_m.
    # x_m[b][c] is the coefficient of block A_bc in X_m.
    # y_m[d][e] is the coefficient of block B_de in Y_m.
    
    # M1: X = A00 + A11. Y = B00 + B11.
    # x1: A00 coef 1, A11 coef 1. Others 0.
    # y1: B00 coef 1, B11 coef 1. Others 0.
    
    x_metas = [
        [[1, 0], [0, 1]], # M1: A00+A11
        [[0, 0], [1, 1]], # M2: A10+A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 1], [0, 0]], # M5: A00+A01
        [[-1, 0], [1, 0]],# M6: A10-A00
        [[0, 1], [0, -1]] # M7: A01-A11
    ]
    
    y_metas = [
        [[1, 0], [0, 1]], # M1: B00+B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 1], [0, -1]],# M3: B01-B11
        [[-1, 0], [1, 0]],# M4: B10-B00
        [[0, 0], [0, 1]], # M5: B11
        [[1, 1], [0, 0]], # M6: B00+B01
        [[0, 0], [1, 1]]  # M7: B10+B11
    ]
    
    # c_metas: Coefficients of M_m in C blocks.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    c_metas = [
        [[1, 0], [0, 1]], # M1 -> C00(1), C11(1)
        [[0, 0], [1, -1]],# M2 -> C10(1), C11(-1)
        [[0, 1], [0, 1]], # M3 -> C01(1), C11(1)
        [[1, 0], [1, 0]], # M4 -> C00(1), C10(1)
        [[-1, 1], [0, 0]],# M5 -> C00(-1), C01(1)
        [[0, 0], [0, 1]], # M6 -> C11(1)
        [[1, 0], [0, 0]]  # M7 -> C00(1)
    ]
    
    # Now generate the 49 terms.
    for m in range(7):
        for s in range(7):
            # u_ms: Coefficients for A
            # For each global A index (i, j) where i,j < 4:
            # i = 2*b + u, j = 2*c + v.
            # Coeff = sum over block coeffs x_m[b][c] * base_u[s][local_idx]
            # where local_idx corresponds to (u, v).
            # local 0:00 (u=0,v=0), 1:01 (u=0,v=1), 2:10 (u=1,v=0), 3:11 (u=1,v=1)
            
            # Initialize u_vec, v_vec, w_vec for this term
            # We will fill U, V, W directly.
            
            for b in range(2):
                for c in range(2):
                    coeff_x = x_metas[m][b][c]
                    if coeff_x == 0: continue
                    for u in range(2):
                        for v in range(2):
                            local_idx = u * 2 + v
                            coeff_u = coeff_x * base_u[s][local_idx]
                            if coeff_u == 0: continue
                            i = 2 * b + u
                            j = 2 * c + v
                            U[i * m + j][col] = coeff_u
                            
            for d in range(2):
                for e in range(2):
                    coeff_y = y_metas[m][d][e]
                    if coeff_y == 0: continue
                    for w in range(2):
                        for x in range(2):
                            local_idx = w * 2 + x
                            coeff_v = coeff_y * base_v[s][local_idx]
                            if coeff_v == 0: continue
                            j = 2 * d + w
                            k = 2 * e + x
                            V[j * p + k][col] = coeff_v
                            
            for d in range(2):
                for e in range(2):
                    coeff_c = c_metas[m][d][e]
                    if coeff_c == 0: continue
                    for w in range(2):
                        for x in range(2):
                            local_idx = w * 2 + x
                            coeff_w = coeff_c * base_w[s][local_idx]
                            if coeff_w == 0: continue
                            k = 2 * d + w
                            i = 2 * e + x
                            W[k * n + i][col] = coeff_w
                            
            col += 1
            
    # Part 3: A[0:4, 0:4] * B[0:4, 4]
    # C[i, 4] += sum_j A[i, j] * B[j, 4]
    # Rank 16.
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1 # k=4
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
