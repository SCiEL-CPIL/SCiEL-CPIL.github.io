"""Direct-sum decomposition: 3x5x7 = 3x2x7 (schoolbook) + 3x3x7 (Strassen core)."""
DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = 3, 5, 7
    m1, m2 = 2, 3  # partition of the 5 inner-dim
    
    rank1 = n * m1 * p  # 42
    rank2 = 5 * p       # 35 (5 Strassen mults x 7 outputs)
    rank = rank1 + rank2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: schoolbook for A[0:3, 0:2] * B[0:2, 0:7]
    for i in range(n):
        for j in range(m1):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Block 2: Strassen for A[0:3, 2:5] * B[2:5, 0:7]
    # A-block: 3x3, B-block: 3x7
    # Strassen 3x3x3 core, applied per output column
    
    # Strassen multipliers for 3x3:
    # M1 = (A00+A22)*(B00+B22)
    # M2 = (A20+A22)*B00
    # M3 = A00*(B10-B20)
    # M4 = A22*(B00-B10)
    # M5 = (A00+A01)*B20
    # C00 = M1 + M3 - M4 - M5
    # C01 = M5 + M2
    # C02 = M5 + M2  (but B02=B12=B22=0 for 3x1, so C02=0)
    # C10 = M1 - M2 + M4
    # C11 = M5 + M2
    # C12 = 0
    # C20 = M1 - M2 + M4
    # C21 = M1 - M3 + M2
    # C22 = 0
    
    # For 3x3 * 3x1, only C0, C1, C2 matter (first column of result)
    # C0 = M1 + M3 - M4 - M5
    # C1 = M5 + M2
    # C2 = M1 - M2 + M4
    
    strassen_mults = [
        # (A_rows_coeffs, B_rows_coeffs, C_row_coeffs)
        # Each is a list of (index, coeff)
        ([((0, 1), (2, 1))], [((0, 1), (2, 1))], [((0, 1), (2, 1))]),   # M1
        ([((2, 1), (2, 1))], [((0, 1))], [((1, 1), (2, -1))]),            # M2
        ([((0, 1))], [((1, 1), (2, -1))], [((0, 1), (2, -1))]),           # M3
        ([((2, 1))], [((0, 1), (1, -1))], [((0, -1), (2, 1))]),           # M4
        ([((0, 1), (0, 1))], [((2, 1))], [((0, -1), (1, 1))]),            # M5
    ]
    
    # Fix: A00+A01 means A[0,0]+A[0,1] in the 3x3 subblock, i.e., A[0,2]+A[0,3] in full
    # Let me redo with proper indices.
    # In the 3x3 subblock A_sub[i,j] = A[i, 2+j] for i,j in 0..2
    # B_sub[j,k] = B[2+j, k] for j in 0..2, k in 0..6
    
    # M1 = (A_sub[0,0]+A_sub[2,2]) * (B_sub[0,k]+B_sub[2,k])
    #     = (A[0,2]+A[2,4]) * (B[2,k]+B[4,k])
    # M2 = (A_sub[2,0]+A_sub[2,2]) * B_sub[0,k]
    #     = (A[2,2]+A[2,4]) * B[2,k]
    # M3 = A_sub[0,0] * (B_sub[1,k]-B_sub[2,k])
    #     = A[0,2] * (B[3,k]-B[4,k])
    # M4 = A_sub[2,2] * (B_sub[0,k]-B_sub[1,k])
    #     = A[2,4] * (B[2,k]-B[3,k])
    # M5 = (A_sub[0,0]+A_sub[0,1]) * B_sub[2,k]
    #     = (A[0,2]+A[0,3]) * B[4,k]
    
    # C_sub[0,k] = M1 + M3 - M4 - M5
    # C_sub[1,k] = M5 + M2
    # C_sub[2,k] = M1 - M2 + M4
    
    strassen_data = [
        # M1
        ([(0, 2, 1), (2, 4, 1)], [(2, 1), (4, 1)], [(0, 1), (2, 1)]),
        # M2
        ([(2, 2, 1), (2, 4, 1)], [(2, 1)], [(1, 1), (2, -1)]),
        # M3
        ([(0, 2, 1)], [(3, 1), (4, -1)], [(0, 1), (2, -1)]),
        # M4
        ([(2, 4, 1)], [(2, 1), (3, -1)], [(0, -1), (2, 1)]),
        # M5
        ([(0, 2, 1), (0, 3, 1)], [(4, 1)], [(0, -1), (1, 1)]),
    ]
    
    for a_coeffs, b_coeffs, c_coeffs in strassen_data:
        for k in range(p):
            for (ai, aj, ac) in a_coeffs:
                U[ai * m + aj][col] += ac
            for (bj, bc) in b_coeffs:
                V[bj * p + k][col] += bc
            for (ci, cc) in c_coeffs:
                W[k * n + ci][col] += cc
            col += 1
    
    return {"U": U, "V": V, "W": W}
