"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition of rank 7 for 2x2x2 matrix multiplication
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U rows indexed by (i,j) for A[i][j], flattened as i*m+j
    # V rows indexed by (j,k) for B[j][k], flattened as j*p+k
    # W rows indexed by (k,i) for C output, flattened as k*n+i (trace convention W[k*n+i])
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # U: for each factor c, U[i*m+j][c] gives the coefficient of A[i][j] in the i-th linear form
    # V: for each factor c, V[j*p+k][c] gives the coefficient of B[j][k] in the j-th linear form
    # W: for each factor c, W[k*n+i][c] gives the coefficient of C[i][k] that receives the product
    
    # Factor 0: M1 = (A00 + A11)(B00 + B11)
    # A linear form: A00 + A11 -> U[0][0]=1, U[3][0]=1
    # B linear form: B00 + B11 -> V[0][0]=1, V[3][0]=1
    # Output: C00 += M1, C11 += M1
    # W[k*n+i]: C00 -> i=0,k=0 -> W[0*2+0]=W[0]; C11 -> i=1,k=1 -> W[1*2+1]=W[3]
    U[0][0] = 1
    U[3][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    W[0][0] = 1   # C00
    W[3][0] = 1   # C11
    
    # Factor 1: M2 = (A10 + A11)B00
    # A linear form: A10 + A11 -> U[2][1]=1, U[3][1]=1
    # B linear form: B00 -> V[0][1]=1
    # Output: C10 += M2
    # C10 -> i=1,k=0 -> W[0*2+1]=W[1]
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1   # C10
    
    # Factor 2: M3 = A00(B10 - B11)
    # A linear form: A00 -> U[0][2]=1
    # B linear form: B10 - B11 -> V[2][2]=1, V[3][2]=-1
    # Output: C01 += M3
    # C01 -> i=0,k=1 -> W[1*2+0]=W[2]
    U[0][2] = 1
    V[2][2] = 1
    V[3][2] = -1
    W[2][2] = 1   # C01
    
    # Factor 3: M4 = A11(B01 - B00)
    # A linear form: A11 -> U[3][3]=1
    # B linear form: B01 - B00 -> V[1][3]=1, V[0][3]=-1
    # Output: C11 += M4
    # C11 -> i=1,k=1 -> W[1*2+1]=W[3]
    U[3][3] = 1
    V[1][3] = 1
    V[0][3] = -1
    W[3][3] = 1   # C11
    
    # Factor 4: M5 = (A00 + A01)B11
    # A linear form: A00 + A01 -> U[0][4]=1, U[1][4]=1
    # B linear form: B11 -> V[3][4]=1
    # Output: C01 += M5
    # C01 -> i=0,k=1 -> W[1*2+0]=W[2]
    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1   # C01
    
    # Factor 5: M6 = (A10 - A00)(B00 + B01)
    # A linear form: A10 - A00 -> U[2][5]=1, U[0][5]=-1
    # B linear form: B00 + B01 -> V[0][5]=1, V[1][5]=1
    # Output: C10 += M6
    # C10 -> i=1,k=0 -> W[0*2+1]=W[1]
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    W[1][5] = 1   # C10
    
    # Factor 6: M7 = (A01 - A11)(B10 + B11)
    # A linear form: A01 - A11 -> U[1][6]=1, U[3][6]=-1
    # B linear form: B10 + B11 -> V[2][6]=1, V[3][6]=1
    # Output: C00 += M7
    # C00 -> i=0,k=0 -> W[0*2+0]=W[0]
    U[1][6] = 1
    U[3][6] = -1
    V[2][6] = 1
    V[3][6] = 1
    W[0][6] = 1   # C00
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

