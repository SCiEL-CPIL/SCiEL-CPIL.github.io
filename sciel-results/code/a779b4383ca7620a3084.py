import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # 1. Define 2x2 Strassen coefficients
    # U2, V2, W2 are 4x7 matrices
    # Indices: 0:00, 1:01, 2:10, 3:11
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Term 0: P1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Term 1: P2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1
    
    # Term 2: P3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1
    
    # Term 3: P4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # Term 4: P5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1; W2[3][4] = -1
    
    # Term 5: P6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Term 6: P7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = 1

    # 2. Define Outer Strassen Block Coefficients
    # CA[t][i][j] is coeff of A_block(i,j) in L^A_t
    # CB[t][i][j] is coeff of B_block(i,j) in L^B_t
    # t in 0..6, i,j in 0..1
    CA = [[[[0]*2 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    CB = [[[[0]*2 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    
    # t=0: A00+A11, B00+B11
    CA[0][0][0] = 1; CA[0][1][1] = 1
    CB[0][0][0] = 1; CB[0][1][1] = 1
    
    # t=1: A10+A11, B00
    CA[1][1][0] = 1; CA[1][1][1] = 1
    CB[1][0][0] = 1
    
    # t=2: A00, B01-B11
    CA[2][0][0] = 1
    CB[2][0][1] = 1; CB[2][1][1] = -1
    
    # t=3: A11, B10-B00
    CA[3][1][1] = 1
    CB[3][1][0] = 1; CB[3][0][0] = -1
    
    # t=4: A00+A01, B11
    CA[4][0][0] = 1; CA[4][0][1] = 1
    CB[4][1][1] = 1
    
    # t=5: A10-A00, B00+B01
    CA[5][1][0] = 1; CA[5][0][0] = -1
    CB[5][0][0] = 1; CB[5][0][1] = 1
    
    # t=6: A01-A11, B10+B11
    CA[6][0][1] = 1; CA[6][1][1] = -1
    CB[6][1][0] = 1; CB[6][1][1] = 1

    # 3. Define Outer Reconstruction Coefficients
    # CC[t][i][j] is coeff of M_t in C_block(i,j)
    CC = [[[[0]*2 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    
    # C00 = M0 + M3 - M4 + M6
    CC[0][0][0] = 1; CC[3][0][0] = 1; CC[4][0][0] = -1; CC[6][0][0] = 1
    
    # C01 = M2 + M4
    CC[2][0][1] = 1; CC[4][0][1] = 1
    
    # C10 = M1 + M3
    CC[1][1][0] = 1; CC[3][1][0] = 1
    
    # C11 = M0 - M1 + M2 + M5
    CC[0][1][1] = 1; CC[1][1][1] = -1; CC[2][1][1] = 1; CC[5][1][1] = 1

    # 4. Construct U, V, W
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for t in range(7):
        for c in range(7):
            col = t * 7 + c
            
            # U contributions
            for p in range(2): # inner row of block
                for q in range(2): # inner col of block
                    u_coeff = U2[2*p+q][c]
                    if u_coeff != 0:
                        for i in range(2): # block row
                            for j in range(2): # block col
                                ca_coeff = CA[t][i][j]
                                if ca_coeff != 0:
                                    global_idx = (2*i+p)*4 + (2*j+q)
                                    U[global_idx][col] += u_coeff * ca_coeff
                                    
            # V contributions
            for q in range(2): # inner row of block (col of A, row of B)
                for r in range(2): # inner col of block
                    v_coeff = V2[2*q+r][c]
                    if v_coeff != 0:
                        for i in range(2): # block row
                            for j in range(2): # block col
                                cb_coeff = CB[t][i][j]
                                if cb_coeff != 0:
                                    global_idx = (2*i+q)*4 + (2*j+r)
                                    V[global_idx][col] += v_coeff * cb_coeff
                                    
            # W contributions
            for p in range(2): # inner row of M_t
                for r in range(2): # inner col of M_t
                    w_coeff = W2[2*r+p][c]
                    if w_coeff != 0:
                        for i in range(2): # block row of C
                            for j in range(2): # block col of C
                                cc_coeff = CC[t][i][j]
                                if cc_coeff != 0:
                                    # W index: k*4 + i_global
                                    # k is col index of C, i is row index of C
                                    # C[2i+p, 2j+r]
                                    # k = 2j+r, i_global = 2i+p
                                    global_idx = (2*j+r)*4 + (2*i+p)
                                    W[global_idx][col] += w_coeff * cc_coeff
                                    
    return {"U": U, "V": V, "W": W}
