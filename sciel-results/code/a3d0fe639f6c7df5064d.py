"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 2x2x6
    # Decompose B into three 2x2 blocks: B0, B1, B2
    # Use Strassen for each: 7 multiplications each
    # Total rank: 21
    n, m, p = DIMENSIONS
    rank = 21
    
    # U: (n*m, rank) = (4, 21)
    # V: (m*p, rank) = (12, 21)
    # W: (p*n, rank) = (12, 21)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B block k: 4k:(0,0), 4k+1:(0,1), 4k+2:(1,0), 4k+3:(1,1)
    # Indices for C block k: 4k:(0,0), 4k+1:(0,1), 4k+2:(1,0), 4k+3:(1,1)
    # Note: W index is k*n + i. For 2x2, C is 2x2.
    # C is 2x6. So C has 12 entries.
    # C[i, 2k+j] corresponds to block k, entry (i,j).
    # W index: (2k+j)*2 + i? No.
    # Task says: W[k*n+i, r].
    # k is column index of B? No, k is index in p.
    # B is 2x6. j is row (0,1), k is col (0..5).
    # C is 2x6. i is row (0,1), k is col (0..5).
    # W index: k*n + i.
    # So W[0, r] is C[0,0], W[1, r] is C[0,1], W[2, r] is C[1,0], W[3, r] is C[1,1] for k=0?
    # No, k is 0..5.
    # W[k*n + i, r].
    # For k=0: W[0, r]=C[0,0], W[1, r]=C[1,0]?
    # n=2.
    # k=0: W[0, r]=C[0,0], W[1, r]=C[1,0].
    # k=1: W[2, r]=C[0,1], W[3, r]=C[1,1].
    # So W index is k*2 + i.
    
    # Strassen terms for block k:
    # M1_k: A=(a00+a11), B=(b00+b11)
    # M2_k: A=(a10+a11), B=b00
    # M3_k: A=a00, B=(b01-b11)
    # M4_k: A=a11, B=(b10-b00)
    # M5_k: A=(a00+a01), B=b11
    # M6_k: A=(a10-a00), B=(b00+b01)
    # M7_k: A=(a01-a11), B=(b10+b11)
    
    # Contributions to C:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's fill U, V, W
    # U index: i*m + j. i=0,1; j=0,1.
    # U[0]=a00, U[1]=a01, U[2]=a10, U[3]=a11.
    
    # V index: j*p + k. j=0,1; k=0..5.
    # V[0]=b00, V[1]=b01, V[2]=b02, V[3]=b03, V[4]=b04, V[5]=b05
    # V[6]=b10, V[7]=b11, V[8]=b12, V[9]=b13, V[10]=b14, V[11]=b15
    
    # W index: k*n + i. k=0..5; i=0,1.
    # W[0]=c00, W[1]=c10
    # W[2]=c01, W[3]=c11
    # ...
    
    # Block 0 (k=0,1): B cols 0,1.
    # B00=V[0], B01=V[1], B10=V[6], B11=V[7]
    # C00=W[0], C01=W[2], C10=W[1], C11=W[3]
    
    # Block 1 (k=2,3): B cols 2,3.
    # B00=V[2], B01=V[3], B10=V[8], B11=V[9]
    # C00=W[4], C01=W[6], C10=W[5], C11=W[7]
    
    # Block 2 (k=4,5): B cols 4,5.
    # B00=V[4], B01=V[5], B10=V[10], B11=V[11]
    # C00=W[8], C01=W[10], C10=W[9], C11=W[11]
    
    col = 0
    for block in range(3):
        # B indices
        b00 = 4 * block
        b01 = 4 * block + 1
        b10 = 6 + 4 * block
        b11 = 7 + 4 * block
        
        # C indices
        c00 = 2 * (2 * block)
        c01 = 2 * (2 * block + 1)
        c10 = 2 * (2 * block) + 1
        c11 = 2 * (2 * block + 1) + 1
        
        # M1: A=(a00+a11), B=(b00+b11)
        U[0][col] = 1
        U[3][col] = 1
        V[b00][col] = 1
        V[b11][col] = 1
        W[c00][col] = 1
        W[c11][col] = 1
        col += 1
        
        # M2: A=(a10+a11), B=b00
        U[2][col] = 1
        U[3][col] = 1
        V[b00][col] = 1
        W[c10][col] = 1
        W[c11][col] = -1
        col += 1
        
        # M3: A=a00, B=(b01-b11)
        U[0][col] = 1
        V[b01][col] = 1
        V[b11][col] = -1
        W[c01][col] = 1
        W[c11][col] = 1
        col += 1
        
        # M4: A=a11, B=(b10-b00)
        U[3][col] = 1
        V[b10][col] = 1
        V[b00][col] = -1
        W[c00][col] = 1
        W[c10][col] = 1
        col += 1
        
        # M5: A=(a00+a01), B=b11
        U[0][col] = 1
        U[1][col] = 1
        V[b11][col] = 1
        W[c00][col] = -1
        W[c01][col] = 1
        col += 1
        
        # M6: A=(a10-a00), B=(b00+b01)
        U[2][col] = 1
        U[0][col] = -1
        V[b00][col] = 1
        V[b01][col] = 1
        W[c11][col] = 1
        col += 1
        
        # M7: A=(a01-a11), B=(b10+b11)
        U[1][col] = 1
        U[3][col] = -1
        V[b10][col] = 1
        V[b11][col] = 1
        W[c00][col] = 1
        W[c11][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
