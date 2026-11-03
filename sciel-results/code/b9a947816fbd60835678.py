"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # P1 = (a11+a22)(b11+b22)
    # P2 = (a21+a22)b11
    # P3 = a11(b12-b22)
    # P4 = a22(b21-b11)
    # P5 = (a11+a12)b22
    # P6 = (a21-a11)(b11+b12)
    # P7 = (a12-a22)(b21+b22)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # We need to map local 2x2 indices to global indices
    # Global A is 2x4. Local A is 2x2.
    # Global B is 4x6. Local B is 2x2.
    # Global C is 2x6. Local C is 2x2.
    
    # Define the 6 block multiplications
    # (a_row_offset, a_col_offset, b_row_offset, b_col_offset)
    blocks = [
        (0, 0, 0, 0), # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        (0, 2, 2, 0), # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        (0, 0, 0, 2), # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        (0, 2, 2, 2), # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        (0, 0, 0, 4), # A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
        (0, 2, 2, 4), # A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    ]
    
    # Strassen terms: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: 2x2 matrix of coefficients for A
    # v_coeffs: 2x2 matrix of coefficients for B
    # w_coeffs: 2x2 matrix of coefficients for C
    
    # P1
    u1 = [[1, 0], [0, 1]]
    v1 = [[1, 0], [0, 1]]
    w1 = [[1, 0], [0, 1]]
    
    # P2
    u2 = [[0, 0], [1, 1]]
    v2 = [[1, 0], [0, 0]]
    w2 = [[0, 0], [1, 0]]
    
    # P3
    u3 = [[1, 0], [0, 0]]
    v3 = [[0, 1], [0, -1]]
    w3 = [[0, 1], [0, 0]]
    
    # P4
    u4 = [[0, 0], [0, 1]]
    v4 = [[0, 0], [1, -1]]
    w4 = [[1, 0], [1, 0]]
    
    # P5
    u5 = [[1, 1], [0, 0]]
    v5 = [[0, 0], [0, 1]]
    w5 = [[-1, 0], [0, 0]]
    
    # P6
    u6 = [[-1, 0], [1, 0]]
    v6 = [[1, 1], [0, 0]]
    w6 = [[0, 0], [0, 1]]
    
    # P7
    u7 = [[0, 1], [0, -1]]
    v7 = [[0, 0], [1, 1]]
    w7 = [[1, 0], [0, 0]]
    
    terms = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    col = 0
    for a_r, a_c, b_r, b_c in blocks:
        for u, v, w in terms:
            for i in range(2):
                for j in range(2):
                    U[(a_r + i) * m + (a_c + j)][col] = u[i][j]
                    V[(a_c + j) * p + (b_c + j)][col] = v[i][j]
                    W[(b_c + j) * n + (a_r + i)][col] = w[i][j]
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
