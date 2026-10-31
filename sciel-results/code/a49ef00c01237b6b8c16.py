"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based rank-49 decomposition for 4x4 matrix multiplication.
    # This uses the Kronecker product of the rank-7 2x2 Strassen decomposition.
    
    # Define the rank-7 decomposition for 2x2 matrices.
    # Indices for 2x2:
    # A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W (C): index k*n + i. n=2.
    # k=0, i=0 -> 0 (C00)
    # k=0, i=1 -> 1 (C10)
    # k=1, i=0 -> 2 (C01)
    # k=1, i=1 -> 3 (C11)
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    # U2: A00(0), A11(3)
    U2[0][0] = 1
    U2[3][0] = 1
    # V2: B00(0), B11(3)
    V2[0][0] = 1
    V2[3][0] = 1
    # W2: C00(0)=1, C11(3)=1
    W2[0][0] = 1
    W2[3][0] = 1
    
    # M2 = (A10 + A11)B00
    # U2: A10(2), A11(3)
    U2[2][1] = 1
    U2[3][1] = 1
    # V2: B00(0)
    V2[0][1] = 1
    # W2: C10(1)=1, C11(3)=-1
    W2[1][1] = 1
    W2[3][1] = -1
    
    # M3 = A00(B01 - B11)
    # U2: A00(0)
    U2[0][2] = 1
    # V2: B01(1), B11(3)
    V2[1][2] = 1
    V2[3][2] = -1
    # W2: C01(2)=1, C11(3)=1
    W2[2][2] = 1
    W2[3][2] = 1
    
    # M4 = A11(B10 - B00)
    # U2: A11(3)
    U2[3][3] = 1
    # V2: B10(2), B00(0)
    V2[2][3] = 1
    V2[0][3] = -1
    # W2: C00(0)=1, C10(1)=1
    W2[0][3] = 1
    W2[1][3] = 1
    
    # M5 = (A00 + A01)B11
    # U2: A00(0), A01(1)
    U2[0][4] = 1
    U2[1][4] = 1
    # V2: B11(3)
    V2[3][4] = 1
    # W2: C00(0)=-1, C01(2)=1
    W2[0][4] = -1
    W2[2][4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    # U2: A10(2), A00(0)
    U2[2][5] = 1
    U2[0][5] = -1
    # V2: B00(0), B01(1)
    V2[0][5] = 1
    V2[1][5] = 1
    # W2: C11(3)=1
    W2[3][5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    # U2: A01(1), A11(3)
    U2[1][6] = 1
    U2[3][6] = -1
    # V2: B10(2), B11(3)
    V2[2][6] = 1
    V2[3][6] = 1
    # W2: C00(0)=1
    W2[0][6] = 1
    
    # Construct 4x4 decomposition
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over rank components
    for r2 in range(7):
        for rsub in range(7):
            r = r2 * 7 + rsub
            
            # Iterate over all indices i, j, k
            for i in range(n):
                I = i // 2
                isub = i % 2
                for j in range(m):
                    J = j // 2
                    jsub = j % 2
                    for k in range(p):
                        K = k // 2
                        ksub = k % 2
                        
                        # U index: i*m + j
                        u_val = U2[I * 2 + J][r2] * U2[isub * 2 + jsub][rsub]
                        if u_val != 0:
                            U[i * m + j][r] = u_val
                            
                        # V index: j*p + k
                        v_val = V2[J * 2 + K][r2] * V2[jsub * 2 + ksub][rsub]
                        if v_val != 0:
                            V[j * p + k][r] = v_val
                            
                        # W index: k*n + i
                        w_val = W2[K * 2 + I][r2] * W2[ksub * 2 + isub][rsub]
                        if w_val != 0:
                            W[k * n + i][r] = w_val
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
