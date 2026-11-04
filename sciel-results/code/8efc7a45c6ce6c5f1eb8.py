"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    # We decompose into 4 independent 2x2x2x2 matrix multiplications
    # Each uses Strassen's 7-multiplication algorithm
    # Total rank = 4 * 7 = 28
    
    # Strassen's 7 rank-1 terms for 2x2 * 2x2 -> 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # where:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # For each 2x2x2x2 sub-problem, define the 7 Strassen terms
    # U factors (for A, shape 2x2 per sub-problem):
    # term 0: [1, 0, 0, 1]  -> a11 + a22
    # term 1: [0, 0, 1, 1]  -> a21 + a22
    # term 2: [1, 0, 0, 0]  -> a11
    # term 3: [0, 0, 0, 1]  -> a22
    # term 4: [1, 1, 0, 0]  -> a11 + a12
    # term 5: [-1, 0, 1, 0] -> a21 - a11
    # term 6: [0, 1, 0, -1] -> a12 - a22
    
    # V factors (for B, shape 2x2 per sub-problem):
    # term 0: [1, 1, 1, 1]  -> b11 + b22 (flattened: b11, b12, b21, b22)
    # Wait, B is 2x2, flattened as [b11, b12, b21, b22]
    # term 0: b11 + b22 -> [1, 0, 0, 1]
    # term 1: b11 -> [1, 0, 0, 0]
    # term 2: b12 - b22 -> [0, 1, 0, -1]
    # term 3: b21 - b11 -> [-1, 0, 1, 0]
    # term 4: b22 -> [0, 0, 0, 1]
    # term 5: b11 + b12 -> [1, 1, 0, 0]
    # term 6: b21 + b22 -> [0, 0, 1, 1]
    
    # W factors (for C, shape 2x2 per sub-problem):
    # C flattened as [c11, c12, c21, c22]
    # term 0: M1 -> c11 += 1, c22 += 1 -> [1, 0, 0, 1]
    # term 1: M2 -> c21 += 1, c22 -= 1 -> [0, 0, 1, -1]
    # term 2: M3 -> c12 += 1, c22 += 1 -> [0, 1, 0, 1]
    # term 3: M4 -> c11 += 1, c21 += 1 -> [1, 0, 1, 0]
    # term 4: M5 -> c11 -= 1, c12 += 1 -> [-1, 1, 0, 0]
    # term 5: M6 -> c22 += 1 -> [0, 0, 0, 1]
    # term 6: M7 -> c11 += 1 -> [1, 0, 0, 0]
    
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    V_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    W_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 2*4 = 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 4*4 = 16 rows
    W = [[0] * rank for _ in range(p * n)]  # 4*2 = 8 rows
    
    # 4 sub-problems:
    # Sub-problem s=0: A[:,0:2] x B[0:2,0:2] -> C[:,0:2]
    # Sub-problem s=1: A[:,2:4] x B[2:4,0:2] -> C[:,0:2]
    # Sub-problem s=2: A[:,0:2] x B[0:2,2:4] -> C[:,2:4]
    # Sub-problem s=3: A[:,2:4] x B[2:4,2:4] -> C[:,2:4]
    
    for s in range(4):
        a_col_start = (s % 2) * 2  # 0 or 2 (which columns of A)
        b_row_start = (s % 2) * 2  # 0 or 2 (which rows of B)
        c_col_start = (s // 2) * 2  # 0 or 2 (which columns of C)
        
        for t in range(7):
            col = s * 7 + t
            
            # U: A is 2x4, rows 0,1; columns a_col_start, a_col_start+1
            # A[i, a_col_start + j] for i in 0,1; j in 0,1
            # U index: i * m + (a_col_start + j) = i * 4 + a_col_start + j
            for i in range(2):
                for j in range(2):
                    U[i * 4 + a_col_start + j][col] = U_strassen[t][i * 2 + j]
            
            # V: B is 4x4, rows b_row_start, b_row_start+1; columns 0,1,2,3
            # But we only use columns c_col_start, c_col_start+1
            # B[b_row_start + i, c_col_start + j] for i in 0,1; j in 0,1
            # V index: (b_row_start + i) * 4 + (c_col_start + j)
            for i in range(2):
                for j in range(2):
                    V[(b_row_start + i) * 4 + c_col_start + j][col] = V_strassen[t][i * 2 + j]
            
            # W: C is 2x4, rows 0,1; columns c_col_start, c_col_start+1
            # C[i, c_col_start + j] for i in 0,1; j in 0,1
            # W index: k * n + i where k is column of C, i is row of C
            # W[k * n + i] = W[(c_col_start + j) * 2 + i]
            for i in range(2):
                for j in range(2):
                    W[(c_col_start + j) * 2 + i][col] = W_strassen[t][i * 2 + j]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
