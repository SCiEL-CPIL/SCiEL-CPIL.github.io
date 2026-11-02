"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 coefficients
    # A indices: 0=a, 1=b, 2=c, 3=d
    # B indices: 0=e, 1=f, 2=g, 3=h
    # C indices: 0=ae+bg, 1=af+bh, 2=ce+dg, 3=cf+dh
    
    # M1: (a+d)(e+h)
    A1 = [1, 0, 0, 1]
    B1 = [1, 0, 0, 1]
    
    # M2: (c+d)e
    A2 = [0, 0, 1, 1]
    B2 = [1, 0, 0, 0]
    
    # M3: a(f-h)
    A3 = [1, 0, 0, 0]
    B3 = [0, 1, 0, -1]
    
    # M4: d(g-e)
    A4 = [0, 0, 0, 1]
    B4 = [0, 0, 1, -1]
    
    # M5: (a+b)h
    A5 = [1, 1, 0, 0]
    B5 = [0, 0, 0, 1]
    
    # M6: (c-a)(e+f)
    A6 = [0, 0, 1, -1]
    B6 = [1, 1, 0, 0]
    
    # M7: (b-d)(g+h)
    A7 = [0, 1, 0, -1]
    B7 = [0, 0, 1, 1]
    
    A_coeffs = [A1, A2, A3, A4, A5, A6, A7]
    B_coeffs = [B1, B2, B3, B4, B5, B6, B7]
    
    # C reconstruction coefficients
    # C0 (top-left) = M1 + M4 - M5 + M7
    C0 = [1, 0, 0, 1, -1, 0, 1]
    # C1 (top-right) = M3 + M5
    C1 = [0, 0, 1, 0, 1, 0, 0]
    # C2 (bottom-left) = M2 + M4
    C2 = [0, 1, 0, 1, 0, 0, 0]
    # C3 (bottom-right) = M1 - M2 + M3 + M6
    C3 = [1, -1, 1, 0, 0, 1, 0]
    
    C_coeffs = [C0, C1, C2, C3]
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    # Part 1: A_L (cols 0,1) * B_T (rows 0,1)
    # A_L indices in U: 0,1,2,3
    # B_T indices in V: 0,1 (row 0), 2,3? No.
    # V is (m*p) = 24.
    # B is 4x6.
    # B_T is rows 0,1.
    # V row 0 corresponds to B[0,0..5]
    # V row 1 corresponds to B[1,0..5]
    
    for j in range(3): # 3 blocks of 2x2
        for k in range(7): # 7 strassen terms
            r = j * 7 + k
            
            # U update
            for i in range(4):
                U[i][r] = A_coeffs[k][i]
            
            # V update
            # Block j is cols 2j, 2j+1
            # B_T block entries:
            # top-left: V[0, 2j]
            # top-right: V[0, 2j+1]
            # bottom-left: V[1, 2j]
            # bottom-right: V[1, 2j+1]
            
            V[0][2*j] += B_coeffs[k][0]
            V[0][2*j+1] += B_coeffs[k][1]
            V[1][2*j] += B_coeffs[k][2]
            V[1][2*j+1] += B_coeffs[k][3]
            
            # W update
            # C is 2x6.
            # Block j contributes to cols 2j, 2j+1
            # C[0, 2j] gets C0
            # C[0, 2j+1] gets C1
            # C[1, 2j] gets C2
            # C[1, 2j+1] gets C3
            
            # W index: k*n + i -> row*6 + col
            W[0*6 + 2*j] += C_coeffs[0][k]
            W[0*6 + 2*j+1] += C_coeffs[1][k]
            W[1*6 + 2*j] += C_coeffs[2][k]
            W[1*6 + 2*j+1] += C_coeffs[3][k]

    # Part 2: A_R (cols 2,3) * B_B (rows 2,3)
    # A_R indices in U: 4,5,6,7
    # B_B indices in V: 2,3 (rows 2,3 of B)
    # V row 2 corresponds to B[2,0..5]
    # V row 3 corresponds to B[3,0..5]
    
    for j in range(3):
        for k in range(7):
            r = 21 + j * 7 + k
            
            # U update
            for i in range(4):
                U[4+i][r] = A_coeffs[k][i]
            
            # V update
            V[2][2*j] += B_coeffs[k][0]
            V[2][2*j+1] += B_coeffs[k][1]
            V[3][2*j] += B_coeffs[k][2]
            V[3][2*j+1] += B_coeffs[k][3]
            
            # W update (same as Part 1)
            W[0*6 + 2*j] += C_coeffs[0][k]
            W[0*6 + 2*j+1] += C_coeffs[1][k]
            W[1*6 + 2*j] += C_coeffs[2][k]
            W[1*6 + 2*j+1] += C_coeffs[3][k]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
