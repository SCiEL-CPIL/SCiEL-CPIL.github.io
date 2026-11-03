"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=9
    # We use a rank-45 decomposition based on 3 parallel Strassen 3x3x3 blocks.
    rank = 45
    U = [[0] * rank for _ in range(n * m)]  # 6 x 45
    V = [[0] * rank for _ in range(m * p)]  # 27 x 45
    W = [[0] * rank for _ in range(p * n)]  # 18 x 45
    
    # Strassen 3x3x3 coefficients
    # u_s, v_s are length 9 vectors corresponding to A[r,c] and B[r,c] with idx = r*3+c
    # w_s is length 9 vector corresponding to C[r,c] with idx = r*3+c
    # We only need C rows 0 and 1 (idx 0,1,2 and 3,4,5). Row 2 (idx 6,7,8) is ignored.
    
    # Define the 15 products M_1 ... M_15
    # M_1 = (A00+A22)(B00+B22)
    # M_2 = (A10+A20)B00
    # M_3 = A01 B10
    # M_4 = (A00+A01)(B11+B21)
    # M_5 = (A20+A22)(B00+B01)
    # M_6 = (A00-A22)(B21+B22)
    # M_7 = (A02+A22)(B00-B02)
    # M_8 = (A00+A11)(B11-B21)
    # M_9 = (A11+A22)(B22-B12)
    # M_10 = (A02+A12)(B11+B21)
    # M_11 = (A02-A11)(B01+B11)
    # M_12 = (A10+A00)(B00+B10)
    # M_13 = (A10-A01)(B01+B11)
    # M_14 = (A12+A22)(B20+B22)
    # M_15 = (A12-A00)(B02+B20)
    
    # u vectors (coeffs for A)
    # idx: 00,01,02, 10,11,12, 20,21,22
    u = [
        [1, 0, 0, 0, 0, 0, 0, 0, 1],  # M1
        [0, 0, 0, 1, 0, 0, 1, 0, 0],  # M2
        [0, 1, 0, 0, 0, 0, 0, 0, 0],  # M3
        [1, 1, 0, 0, 0, 0, 0, 0, 0],  # M4
        [0, 0, 0, 0, 0, 0, 1, 0, 1],  # M5
        [1, 0, 0, 0, 0, 0, 0, 0, -1], # M6
        [0, 0, 1, 0, 0, 0, 0, 0, 1],  # M7
        [1, 0, 0, 0, 1, 0, 0, 0, 0],  # M8
        [0, 0, 0, 0, 1, 0, 0, 0, 1],  # M9
        [0, 0, 1, 0, 0, 1, 0, 0, 0],  # M10
        [0, 0, 1, 0, -1, 0, 0, 0, 0], # M11
        [1, 0, 0, 1, 0, 0, 0, 0, 0],  # M12
        [0, -1, 0, 1, 0, 0, 0, 0, 0], # M13
        [0, 0, 0, 0, 0, 1, 0, 0, 1],  # M14
        [0, 0, 0, 0, -1, 1, 0, 0, 0]  # M15
    ]
    
    # v vectors (coeffs for B)
    v = [
        [1, 0, 0, 0, 0, 0, 0, 0, 1],  # M1
        [1, 0, 0, 0, 0, 0, 0, 0, 0],  # M2
        [0, 0, 0, 1, 0, 0, 0, 0, 0],  # M3
        [0, 0, 0, 0, 1, 0, 0, 1, 0],  # M4
        [1, 1, 0, 0, 0, 0, 0, 0, 0],  # M5
        [0, 0, 0, 0, 0, 0, 0, 1, 1],  # M6
        [1, 0, -1, 0, 0, 0, 0, 0, 0], # M7
        [0, 0, 0, 0, 1, -1, 0, 0, 0], # M8
        [0, 0, 0, 0, 0, -1, 0, 0, 1], # M9
        [0, 0, 0, 0, 1, 0, 0, 1, 0],  # M10
        [0, 1, 0, 0, 1, 0, 0, 0, 0],  # M11
        [1, 0, 0, 1, 0, 0, 0, 0, 0],  # M12
        [0, 1, 0, 0, 1, 0, 0, 0, 0],  # M13
        [0, 0, 0, 0, 0, 0, 1, 0, 1],  # M14
        [0, 1, 0, 0, 0, 0, 1, 0, 0]   # M15
    ]
    
    # w matrix (coeffs for C)
    # C00, C01, C02, C10, C11, C12, C20, C21, C22
    w = [
        [1, 0, 0, 0, 1, 0, 0, 1, 0],  # M1
        [0, 0, 0, 1, -1, 0, 1, 0, 0], # M2
        [0, 1, 0, 0, 0, 0, 0, -1, 0], # M3
        [-1, 1, 0, 0, 0, 0, 0, 0, 0], # M4
        [0, 0, 0, 0, 1, 0, -1, 0, 1], # M5
        [-1, 0, 0, 1, 0, 0, 0, 0, 0], # M6
        [0, 0, 1, 0, 0, 0, 0, 0, -1], # M7
        [1, 0, 0, 0, 0, 1, 0, 0, 0],  # M8
        [0, 0, 0, 0, -1, 1, 0, 0, 0], # M9
        [0, 0, -1, 0, 0, 0, 0, 0, 0], # M10
        [0, 0, 1, 0, 0, 0, 0, 0, 0],  # M11
        [0, 0, 0, 0, 0, 0, 0, -1, 0], # M12
        [0, 0, 0, 0, 0, 0, 1, 0, 0],  # M13
        [0, 0, 0, 0, 0, 0, 0, 1, -1], # M14
        [0, 0, 0, 0, 0, 0, 0, 0, 1]   # M15
    ]
    
    # Build U, V, W
    # n=2, m=3, p=9
    # U rows: i in 0..1, j in 0..2 -> idx = i*3 + j
    # V rows: j in 0..2, k in 0..8 -> idx = j*9 + k
    # W rows: k in 0..8, i in 0..1 -> idx = k*2 + i
    
    for b in range(3):
        for s in range(15):
            col = b * 15 + s
            # Add contributions to U
            for idx_a in range(9):
                if u[s][idx_a] != 0:
                    r = idx_a // 3
                    c = idx_a % 3
                    if r < 2: # Only rows 0,1 of A are non-zero
                        U[r * 3 + c][col] += u[s][idx_a]
            # Add contributions to V
            for idx_b in range(9):
                if v[s][idx_b] != 0:
                    r = idx_b // 3
                    c = idx_b % 3
                    global_k = b * 3 + c
                    V[r * 9 + global_k][col] += v[s][idx_b]
            # Add contributions to W
            for idx_c in range(9):
                if w[s][idx_c] != 0:
                    r = idx_c // 3
                    c = idx_c % 3
                    if r < 2: # Only rows 0,1 of C are needed
                        global_k = b * 3 + c
                        W[global_k * 2 + r][col] += w[s][idx_c]
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
