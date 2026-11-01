DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen bilinear forms for 2x2 x 2x2
    # P = [[p0,p1],[p2,p3]], Q = [[q0,q1],[q2,q3]]
    # 7 forms with (P-coeffs, Q-coeffs):
    strassen_forms = [
        # (P: [p0,p1,p2,p3], Q: [q0,q1,q2,q3])
        ([1, 0, 0, 1], [1, 0, 0, 1]),  # b0 = (p0+p3)(q0+q3)
        ([0, 0, 1, 1], [1, 0, 0, 0]),  # b1 = (p2+p3)q0
        ([1, 0, 0, 0], [0, 1, 0, 1]),  # b2 = p0(q1+q3)
        ([0, 0, 0, 1], [0, 0, 1, -1]), # b3 = p3(q2-q0)
        ([1, 0, 1, 0], [0, 0, 0, 1]),  # b4 = (p0+p2)q3
        ([0, 1, 0, -1], [0, 0, 1, 1]), # b5 = (p1-p3)(q2+q3)
        ([1, 0, -1, 0], [1, 0, 1, 0]), # b6 = (p0-p2)(q0+q2)
    ]
    
    # Output coefficients: C00, C01, C10, C11 in terms of b0..b6
    # C00 = b0 + b3 - b5 - b6
    # C01 = b2 + b5
    # C10 = b1 + b4
    # C11 = b0 - b1 + b2 + b4
    output_coeffs = [
        [1, 0, 0, 1, 0, -1, -1],  # C00
        [0, 0, 1, 0, 0, 1, 0],    # C01
        [0, 1, 0, 0, 1, 0, 0],    # C10
        [1, -1, 1, 0, 1, 0, 0],   # C11
    ]
    
    # Define the 4 sub-problems
    # Each: (P_row_indices, Q_row_indices, output_row_indices)
    # P_row_indices: [p0_row, p1_row, p2_row, p3_row] in U
    # Q_row_indices: [q0_row, q1_row, q2_row, q3_row] in V
    # output_row_indices: [c00_row, c01_row, c10_row, c11_row] in W
    
    # U row for A[i][k] = i*4 + k
    # V row for B[k][j] = k*4 + j
    # W row for C[i][j] = j*2 + i
    
    subproblems = [
        # A_1 * B_1a → C[0][0], C[0][1], C[1][0], C[1][1]
        {
            'P': [0*4+0, 0*4+1, 1*4+0, 1*4+1],  # A[0][0],A[0][1],A[1][0],A[1][1]
            'Q': [0*4+0, 0*4+1, 1*4+0, 1*4+1],  # B[0][0],B[0][1],B[1][0],B[1][1]
            'out': [0*2+0, 1*2+0, 0*2+1, 1*2+1],  # C[0][0],C[0][1],C[1][0],C[1][1]
        },
        # A_1 * B_1b → C[0][2], C[0][3], C[1][2], C[1][3]
        {
            'P': [0*4+0, 0*4+1, 1*4+0, 1*4+1],  # A[0][0],A[0][1],A[1][0],A[1][1]
            'Q': [0*4+2, 0*4+3, 1*4+2, 1*4+3],  # B[0][2],B[0][3],B[1][2],B[1][3]
            'out': [2*2+0, 3*2+0, 2*2+1, 3*2+1],  # C[0][2],C[0][3],C[1][2],C[1][3]
        },
        # A_2 * B_2a → C[0][0], C[0][1], C[1][0], C[1][1]
        {
            'P': [0*4+2, 0*4+3, 1*4+2, 1*4+3],  # A[0][2],A[0][3],A[1][2],A[1][3]
            'Q': [2*4+0, 2*4+1, 3*4+0, 3*4+1],  # B[2][0],B[2][1],B[3][0],B[3][1]
            'out': [0*2+0, 1*2+0, 0*2+1, 1*2+1],  # C[0][0],C[0][1],C[1][0],C[1][1]
        },
        # A_2 * B_2b → C[0][2], C[0][3], C[1][2], C[1][3]
        {
            'P': [0*4+2, 0*4+3, 1*4+2, 1*4+3],  # A[0][2],A[0][3],A[1][2],A[1][3]
            'Q': [2*4+2, 2*4+3, 3*4+2, 3*4+3],  # B[2][2],B[2][3],B[3][2],B[3][3]
            'out': [2*2+0, 3*2+0, 2*2+1, 3*2+1],  # C[0][2],C[0][3],C[1][2],C[1][3]
        },
    ]
    
    col = 0
    for sp in subproblems:
        for s in range(7):
            p_coeffs, q_coeffs = strassen_forms[s]
            # Fill U
            for pi in range(4):
                if p_coeffs[pi] != 0:
                    U[sp['P'][pi]][col] += p_coeffs[pi]
            # Fill V
            for qi in range(4):
                if q_coeffs[qi] != 0:
                    V[sp['Q'][qi]][col] += q_coeffs[qi]
            # Fill W
            for oi in range(4):
                if output_coeffs[oi][s] != 0:
                    W[sp['out'][oi]][col] += output_coeffs[oi][s]
            col += 1
    
    return {"U": U, "V": V, "W": W}
