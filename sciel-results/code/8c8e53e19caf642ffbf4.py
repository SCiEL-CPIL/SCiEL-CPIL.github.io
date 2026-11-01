"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Strategy: decompose 2x12 multiplication as 4 independent 2x3 multiplications
    # Each 2x3 multiplication uses a rank-5 decomposition (Strassen-like)
    # Total rank = 4 * 5 = 20
    
    # For 2x2 by 2x3 matrix multiplication:
    # A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    #
    # Rank-5 decomposition:
    # We need 5 terms such that the sum of rank-1 tensors equals the multiplication tensor.
    
    # A indices: 0=a, 1=b, 2=c, 3=d
    # B indices: 0=e, 1=f, 2=g, 3=h, 4=i, 5=j
    # C indices: 0=ae+bh, 1=af+bi, 2=ag+bj, 3=ce+dh, 4=cf+di, 5=cg+dj
    
    # Rank-5 decomposition for 2x2 by 2x3:
    # Using the following 5 rank-1 tensors:
    
    # Term 0: (a+c)(e+h) -> contributes to c00 and c10
    # Term 1: (b+d)(f+i) -> contributes to c01 and c11  
    # Term 2: (a+b)(g+j) -> contributes to c02 and c12
    # Term 3: a*e -> contributes to c00
    # Term 4: c*h -> contributes to c10
    
    # Actually, let me use a verified decomposition. The key insight is:
    # c00 = ae + bh
    # c01 = af + bi
    # c02 = ag + bj
    # c10 = ce + dh
    # c11 = cf + di
    # c12 = cg + dj
    
    # We can write:
    # (a+c)(e+h) = ae + ah + ce + ch
    # (a-b)(e-h) = ae - ah - be + bh
    # Sum: 2ae + 2bh + ce + ch - be - ah... this gets complicated.
    
    # Let me use a simpler approach with 5 terms:
    # t1 = (a+c)(e+h)
    # t2 = (a-c)(e-h)  
    # t3 = b*f
    # t4 = d*i
    # t5 = ... hmm
    
    # Actually, the cleanest rank-5 decomposition for 2x2 by 2x3 is:
    # Let me just verify a candidate numerically.
    
    # Candidate rank-5 decomposition:
    A5 = np.array([
        [1, 0, 0, 0],  # a
        [0, 1, 0, 0],  # b
        [0, 0, 1, 0],  # c
        [0, 0, 0, 1],  # d
        [1, 1, 1, 1],  # a+b+c+d
    ], dtype=float)
    
    B5 = np.array([
        [1, 0, 0, 1, 0, 0],  # e+h
        [0, 1, 0, 0, 1, 0],  # f+i
        [1, 0, 0, 1, 0, 0],  # e+h
        [0, 1, 0, 0, 1, 0],  # f+i
        [1, 1, 1, 1, 1, 1],  # e+f+g+h+i+j
    ], dtype=float)
    
    C5 = np.array([
        [1, 0, 0, 1, 0, 0],  # c00 + c10
        [0, 1, 0, 0, 1, 0],  # c01 + c11
        [1, 0, 0, 1, 0, 0],  # c00 + c10
        [0, 1, 0, 0, 1, 0],  # c01 + c11
        [0, 0, 1, 0, 0, 1],  # c02 + c12
    ], dtype=float)
    
    # Verify this decomposition
    T_target = np.zeros((4, 6, 6), dtype=float)
    # c00 = ae + bh
    T_target[0, 0, 0] = 1  # a*e
    T_target[1, 3, 0] = 1  # b*h
    # c01 = af + bi
    T_target[0, 1, 1] = 1  # a*f
    T_target[1, 4, 1] = 1  # b*i
    # c02 = ag + bj
    T_target[0, 2, 2] = 1  # a*g
    T_target[1, 5, 2] = 1  # b*j
    # c10 = ce + dh
    T_target[2, 0, 3] = 1  # c*e
    T_target[3, 3, 3] = 1  # d*h
    # c11 = cf + di
    T_target[2, 1, 4] = 1  # c*f
    T_target[3, 4, 4] = 1  # d*i
    # c12 = cg + dj
    T_target[2, 2, 5] = 1  # c*g
    T_target[3, 5, 5] = 1  # d*j
    
    T_recon = np.zeros((4, 6, 6), dtype=float)
    for r in range(5):
        for i in range(4):
            for j in range(6):
                for k in range(6):
                    T_recon[i, j, k] += A5[r, i] * B5[r, j] * C5[r, k]
    
    if not np.allclose(T_recon, T_target):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Now construct the full 2x2 by 2x12 decomposition
    # B is 2x12, so we split it into 4 blocks of 2x3
    # Block 0: B[:, 0:3], Block 1: B[:, 3:6], Block 2: B[:, 6:9], Block 3: B[:, 9:12]
    
    rank = 4 * 5  # 20
    U = [[0] * rank for _ in range(n * m)]  # 4 x 20
    V = [[0] * rank for _ in range(m * p)]  # 24 x 20
    W = [[0] * rank for _ in range(p * n)]  # 24 x 20
    
    for block in range(4):
        for r in range(5):
            col = block * 5 + r
            
            # U: A indices are the same for all blocks
            for i in range(n * m):
                U[i][col] = A5[r, i]
            
            # V: B indices - block b corresponds to columns 3*b to 3*b+2 of B
            # B row 0: indices 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
            # B row 1: indices 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
            # For block b, B row 0 uses indices 3*b, 3*b+1, 3*b+2
            #           B row 1 uses indices 12+3*b, 12+3*b+1, 12+3*b+2
            for j in range(6):
                b_row = j // 3  # 0 or 1
                b_col = j % 3   # 0, 1, or 2
                v_idx = b_row * 12 + 3 * block + b_col
                V[v_idx][col] = B5[r, j]
            
            # W: C indices - block b corresponds to columns 3*b to 3*b+2 of C
            # C row 0: indices 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
            # C row 1: indices 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
            for k in range(6):
                c_row = k // 3  # 0 or 1
                c_col = k % 3   # 0, 1, or 2
                w_idx = c_row * 12 + 3 * block + c_col
                W[w_idx][col] = C5[r, k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
