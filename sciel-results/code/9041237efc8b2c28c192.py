DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 7
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(b_cols):
        nonlocal col
        k0, k1 = b_cols
        
        # M1: (A00+A11)(B00+B11)
        U[0*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*7+k0][col] = 1
        V[1*7+k1][col] = 1
        W[k0*2+0][col] = 1
        W[k1*2+1][col] = 1
        col += 1
        
        # M2: (A10+A11)B00
        U[1*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*7+k0][col] = 1
        W[k0*2+1][col] = 1
        W[k1*2+1][col] = -1
        col += 1
        
        # M3: A00(B01-B11)
        U[0*3+0][col] = 1
        V[0*7+k1][col] = 1
        V[1*7+k1][col] = -1
        W[k1*2+0][col] = 1
        W[k1*2+1][col] = 1
        col += 1
        
        # M4: A11(B10-B00)
        U[1*3+1][col] = 1
        V[1*7+k0][col] = 1
        V[0*7+k0][col] = -1
        W[k0*2+0][col] = 1
        W[k0*2+1][col] = 1
        col += 1
        
        # M5: (A00+A10)B11
        U[0*3+0][col] = 1
        U[1*3+0][col] = 1
        V[1*7+k1][col] = 1
        W[k0*2+0][col] = -1
        W[k1*2+0][col] = 1
        col += 1
        
        # M6: (A01-A00)(B00+B10)
        U[0*3+1][col] = 1
        U[0*3+0][col] = -1
        V[0*7+k0][col] = 1
        V[1*7+k0][col] = 1
        W[k1*2+1][col] = 1
        col += 1
        
        # M7: (A10-A11)(B01+B11)
        U[1*3+0][col] = 1
        U[1*3+1][col] = -1
        V[0*7+k1][col] = 1
        V[1*7+k1][col] = 1
        W[k0*2+0][col] = 1
        col += 1

    add_strassen_block((0, 1))
    add_strassen_block((2, 3))
    add_strassen_block((4, 5))
    
    # Schoolbook for B col 6
    for i in range(2):
        for j in range(2):
            U[i*3+j][col] = 1
            V[j*7+6][col] = 1
            W[6*2+i][col] = 1
            col += 1
            
    # Cross terms
    for i in range(2):
        for k in range(7):
            U[i*3+2][col] = 1
            V[2*7+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
