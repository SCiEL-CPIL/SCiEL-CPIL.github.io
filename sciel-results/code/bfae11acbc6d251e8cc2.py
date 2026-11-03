"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Kronecker product: 2x4x6 = (2x2x2) ⊗ (1x2x3)
    # n = n1 * n2 = 2 * 1
    # m = m1 * m2 = 2 * 2  
    # p = p1 * p2 = 2 * 3
    
    # Index mapping:
    # i = i1 * n2 + i2 = i1 * 1 + i2, i1 in {0,1}, i2 in {0}
    # j = j1 * m2 + j2 = j1 * 2 + j2, j1 in {0,1}, j2 in {0,1}
    # k = k1 * p2 + k2 = k1 * 3 + k2, k1 in {0,1}, k2 in {0,1,2}
    
    # So: i = i1, j = 2*j1 + j2, k = 3*k1 + k2
    
    rank_total = 7 * 6
    
    U = [[0] * rank_total for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank_total for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank_total for _ in range(p * n)]  # 12 x 42
    
    # Strassen coefficients for 2x2x2
    # A: 2x2, B: 2x2, C: 2x2
    strassen_a = [
        [(0,0,1), (1,1,1)],      # M1
        [(1,0,1), (1,1,1)],      # M2
        [(0,0,1)],                # M3
        [(1,1,1)],                # M4
        [(0,0,1), (0,1,1)],      # M5
        [(1,0,1), (0,0,-1)],     # M6
        [(0,1,1), (1,1,-1)],     # M7
    ]
    
    strassen_b = [
        [(0,0,1), (1,1,1)],      # M1
        [(0,0,1)],                # M2
        [(1,0,1), (1,1,-1)],     # M3
        [(1,1,1), (0,0,-1)],     # M4
        [(1,1,1)],                # M5
        [(0,0,1), (0,1,1)],      # M6
        [(1,0,1), (1,1,1)],      # M7
    ]
    
    strassen_c = [
        [(0,0,1), (0,1,1), (1,0,1), (1,1,1)],  # M1
        [(1,0,1)],                # M2
        [(0,1,1)],                # M3
        [(1,0,1), (1,1,-1)],     # M4
        [(0,0,1)],                # M5
        [(1,1,1)],                # M6
        [(0,1,1), (1,1,1)],      # M7
    ]
    
    # Schoolbook for 1x2x3
    schoolbook_a = []
    schoolbook_b = []
    schoolbook_c = []
    for j2 in range(2):
        for k2 in range(3):
            schoolbook_a.append([(0, j2, 1)])
            schoolbook_b.append([(j2, k2, 1)])
            schoolbook_c.append([(0, k2, 1)])
    
    col = 0
    for s in range(7):
        for t in range(6):
            # U: row index = i * m + j = i * 4 + j
            # i = i1, j = 2*j1 + j2
            for (i1, j1, ca) in strassen_a[s]:
                for (i2, j2, cb) in schoolbook_a[t]:
                    i = i1
                    j = 2 * j1 + j2
                    row_u = i * m + j
                    U[row_u][col] += ca * cb
            
            # V: row index = j * p + k = j * 6 + k
            # j = 2*j1 + j2, k = 3*k1 + k2
            for (j1, k1, cb) in strassen_b[s]:
                for (j2, k2, cc) in schoolbook_b[t]:
                    j = 2 * j1 + j2
                    k = 3 * k1 + k2
                    row_v = j * p + k
                    V[row_v][col] += cb * cc
            
            # W: row index = k * n + i = k * 2 + i
            # i = i1, k = 3*k1 + k2
            for (i1, k1, cc) in strassen_c[s]:
                for (i2, k2, cd) in schoolbook_c[t]:
                    i = i1
                    k = 3 * k1 + k2
                    row_w = k * n + i
                    W[row_w][col] += cc * cd
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
