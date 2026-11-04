"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 114 decomposition: Strassen 4x4x4 (rank 49) + Schoolbook 4x4x5 (rank 80)
    rank = 114
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: Strassen 4x4x4 for k in 0..3 ---
    
    # Define 2x2x2 Strassen factors (rank 7)
    # U22[i,j,r], V22[j,k,r], W22[k,i,r] for i,j,k in 0..1
    U22 = [[0]*7 for _ in range(4)]
    V22 = [[0]*7 for _ in range(4)]
    W22 = [[0]*7 for _ in range(4)]
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U22[0][0] = 1; U22[3][0] = 1 # A00, A11
    V22[0][0] = 1; V22[3][0] = 1 # B00, B11
    W22[0][0] = 1; W22[3][3] = 1 # C00, C11
    
    # r=1: M2 = (A10+A11)B00
    U22[2][1] = 1; U22[3][1] = 1 # A10, A11
    V22[0][1] = 1                # B00
    W22[0][1] = 1; W22[3][1] = -1 # C10 (+1), C11 (-1). W[k,i]. C10->k=0,i=1. C11->k=1,i=1.
    
    # r=2: M3 = (A00+A01)(B01-B11)
    U22[0][2] = 1; U22[1][2] = 1 # A00, A01
    V22[1][2] = 1; V22[3][2] = -1 # B01, B11
    W22[1][0] = 1; W22[3][3] = 1  # C01 (+1), C11 (+1). C01->k=1,i=0. C11->k=1,i=1.
    
    # r=3: M4 = A11(B10-B00)
    U22[3][3] = 1                # A11
    V22[2][3] = 1; V22[0][3] = -1 # B10, B00
    W22[0][0] = 1                # C00 (+1). C00->k=0,i=0.
    
    # r=4: M5 = (A00+A10)B11
    U22[0][4] = 1; U22[2][4] = 1 # A00, A10
    V22[3][4] = 1                # B11
    W22[0][0] = -1; W22[3][3] = 1 # C00 (-1), C11 (+1).
    
    # r=5: M6 = (A11-A00)(B00+B01)
    U22[3][5] = 1; U22[0][5] = -1 # A11, A00
    V22[0][5] = 1; V22[1][5] = 1  # B00, B01
    W22[0][1] = 1                # C10 (+1). C10->k=0,i=1.
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U22[1][6] = 1; U22[3][6] = -1 # A01, A11
    V22[2][6] = 1; V22[3][6] = 1  # B10, B11
    W22[1][0] = 1; W22[0][0] = -1 # C01 (+1), C00 (-1). C01->k=1,i=0. C00->k=0,i=0.

    # Define block coefficients for 4x4x4 Strassen
    # A_coeffs[r][I][J], B_coeffs[r][J][K], C_coeffs[r][I][K]
    # I,J,K in 0..1
    A_coeffs = [[0]*2 for _ in range(2)] # Placeholder, will be list of 7 matrices
    A_coeffs = []
    B_coeffs = []
    C_coeffs = []
    
    # M1
    A_coeffs.append([[1,0],[0,1]]) # A00+A11
    B_coeffs.append([[1,0],[0,1]]) # B00+B11
    C_coeffs.append([[1,0],[0,1]]) # C00+C11
    
    # M2
    A_coeffs.append([[0,0],[1,1]]) # A10+A11
    B_coeffs.append([[1,0],[0,0]]) # B00
    C_coeffs.append([[0,0],[1,-1]]) # C10-C11. C10 is I=1,K=0. C11 is I=1,K=1.
    
    # M3
    A_coeffs.append([[1,1],[0,0]]) # A00+A01
    B_coeffs.append([[0,1],[0,-1]]) # B01-B11
    C_coeffs.append([[0,1],[0,1]]) # C01+C11. C01 is I=0,K=1. C11 is I=1,K=1.
    
    # M4
    A_coeffs.append([[0,0],[0,1]]) # A11
    B_coeffs.append([[0,-1],[1,0]]) # B10-B00. B10 is J=1,K=0. B00 is J=0,K=0.
    C_coeffs.append([[1,0],[0,0]]) # C00
    
    # M5
    A_coeffs.append([[1,0],[1,0]]) # A00+A10
    B_coeffs.append([[0,0],[0,1]]) # B11
    C_coeffs.append([[-1,0],[0,1]]) # -C00+C11
    
    # M6
    A_coeffs.append([[0,0],[0,1]]) # A11-A00? No. A11 is I=1,J=1. A00 is I=0,J=0.
    # Wait, M6 = (A11-A00)(B00+B01).
    # A11 is block (1,1). A00 is block (0,0).
    A_coeffs.append([[-1,0],[0,1]]) # -A00+A11
    B_coeffs.append([[1,1],[0,0]]) # B00+B01. B00 is J=0,K=0. B01 is J=0,K=1.
    C_coeffs.append([[0,0],[1,0]]) # C10. I=1,K=0.
    
    # M7
    A_coeffs.append([[0,1],[0,-1]]) # A01-A11. A01 is I=0,J=1. A11 is I=1,J=1.
    B_coeffs.append([[0,0],[1,1]]) # B10+B11. B10 is J=1,K=0. B11 is J=1,K=1.
    C_coeffs.append([[-1,1],[0,0]]) # -C00+C01. C00 is I=0,K=0. C01 is I=0,K=1.

    col = 0
    for r in range(7):
        for s in range(7):
            # U
            for I in range(2):
                for J in range(2):
                    a_coeff = A_coeffs[r][I][J]
                    if a_coeff != 0:
                        for i2 in range(2):
                            for j2 in range(2):
                                u_val = U22[i2*2+j2][s] * a_coeff
                                if u_val != 0:
                                    i = I*2 + i2
                                    j = J*2 + j2
                                    U[i*m+j][col] += u_val
            # V
            for J in range(2):
                for K in range(2):
                    b_coeff = B_coeffs[r][J][K]
                    if b_coeff != 0:
                        for j2 in range(2):
                            for k2 in range(2):
                                v_val = V22[j2*2+k2][s] * b_coeff
                                if v_val != 0:
                                    j = J*2 + j2
                                    k = K*2 + k2 # Local k 0..3
                                    V[j*p+k][col] += v_val
            # W
            for I in range(2):
                for K in range(2):
                    c_coeff = C_coeffs[r][I][K]
                    if c_coeff != 0:
                        for k2 in range(2):
                            for i2 in range(2):
                                w_val = W22[k2*2+i2][s] * c_coeff
                                if w_val != 0:
                                    k = K*2 + k2 # Local k 0..3
                                    i = I*2 + i2
                                    W[k*n+i][col] += w_val
            col += 1
            
    # --- Part 2: Schoolbook 4x4x5 for k in 4..8 ---
    for i in range(n):
        for j in range(m):
            for k_local in range(5):
                k = 4 + k_local
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
