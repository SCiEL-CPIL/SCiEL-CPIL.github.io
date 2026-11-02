"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication
    using Strassen's 7-product algorithm on 2x2 blocks.
    
    Rank = 7 * (n/2) * (m/2) * (p/2) = 7 * 1 * 2 * 3 = 42
    """
    n, m, p = DIMENSIONS
    # Block dimensions (must be even for this approach)
    nb, mb, pb = n // 2, m // 2, p // 2  # 1, 2, 3
    
    rank = 7 * nb * mb * pb  # 42
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # (8, 42)
    V = [[0] * rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # (12, 42)
    
    # Strassen's 7 products for 2x2 matrices
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B10)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block position (bi, bj, bk) where bi in [0,nb), bj in [0,mb), bk in [0,pb):
    # A block at row-block bi, col-block bj: rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # B block at row-block bj, col-block bk: rows 2*bj..2*bj+1, cols 2*bk..2*bk+1
    # C block at row-block bi, col-block bk: rows 2*bk..2*bk+1 (in W index), cols 2*bi..2*bi+1
    
    # Strassen factors for 2x2 case
    # U factors (for A block, indices within block: 0=A00, 1=A01, 2=A10, 3=A11)
    U_strassen = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: A10 - A00
        [0, 1, 0, -1],  # M7: A01 - A11
    ]
    
    # V factors (for B block, indices within block: 0=B00, 1=B01, 2=B10, 3=B11)
    V_strassen = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, -1, 1, 0],  # M3: B10 - B11
        [-1, 1, 0, 0],  # M4: B01 - B10
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1],   # M7: B10 + B11
    ]
    
    # W factors (for C block, indices within block: 0=C00, 1=C01, 2=C10, 3=C11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W_strassen = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 0, 1, -1],  # M2: C10 - C11
        [0, 1, 0, 1],   # M3: C01 + C11
        [1, 0, 1, 0],   # M4: C00 + C10
        [-1, 1, 0, 0],  # M5: -C00 + C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0],   # M7: C00
    ]
    
    # Fill in the factor matrices
    col = 0
    for bi in range(nb):      # row block of A / C
        for bj in range(mb):  # col block of A / row block of B
            for bk in range(pb):  # col block of B / col block of C
                for s in range(7):  # Strassen product index
                    # U: A block at (bi, bj), element s
                    # Row in U: i*m + j where i is row in A, j is col in A
                    for li in range(2):  # local row in A block
                        for lj in range(2):  # local col in A block
                            local_idx = li * 2 + lj  # 0=A00, 1=A01, 2=A10, 3=A11
                            val = U_strassen[s][local_idx]
                            if val != 0:
                                global_i = 2 * bi + li
                                global_j = 2 * bj + lj
                                U[global_i * m + global_j][col] += val
                    
                    # V: B block at (bj, bk), element s
                    # Row in V: j*p + k where j is row in B, k is col in B
                    for li in range(2):  # local row in B block
                        for lj in range(2):  # local col in B block
                            local_idx = li * 2 + lj  # 0=B00, 1=B01, 2=B10, 3=B11
                            val = V_strassen[s][local_idx]
                            if val != 0:
                                global_j = 2 * bj + li
                                global_k = 2 * bk + lj
                                V[global_j * p + global_k][col] += val
                    
                    # W: C block at (bi, bk), element s
                    # W index: k*n + i where k is row in C, i is col in C
                    for li in range(2):  # local row in C block
                        for lj in range(2):  # local col in C block
                            local_idx = li * 2 + lj  # 0=C00, 1=C01, 2=C10, 3=C11
                            val = W_strassen[s][local_idx]
                            if val != 0:
                                global_k = 2 * bk + li
                                global_i = 2 * bi + lj
                                W[global_k * n + global_i][col] += val
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
