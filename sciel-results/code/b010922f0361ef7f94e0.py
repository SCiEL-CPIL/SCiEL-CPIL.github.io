"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    # Combined Strassen (rank 7) for 2x2x2 + schoolbook (rank 4) for 2x2x1 = rank 11
    n, m, p = DIMENSIONS  # n=2, m=2, p=3
    rank = 11
    
    # U has shape (n*m, rank) = (4, 11)
    # V has shape (m*p, rank) = (6, 11)
    # W has shape (p*n, rank) = (6, 11)
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part: columns 0-6 for the 2x2x2 subproblem (columns 0,1 of B)
    # A is 2x2: A[i,j] with i in {0,1}, j in {0,1}
    # B is 2x2: B[j,k] with j in {0,1}, k in {0,1}
    # C is 2x2: C[i,k] with i in {0,1}, k in {0,1}
    #
    # U index: i*m + j = i*2 + j
    # V index: j*p + k = j*3 + k
    # W index: k*n + i = k*2 + i
    
    # Strassen factors:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A11)(B10+B11)
    # M7 = (A00-A10)(B00+B10)
    
    # Column 0: M1 = (A00+A11)(B00+B11)
    # U: A00 -> U[0*2+0]=U[0], A11 -> U[1*2+1]=U[3]
    U[0][0] = 1
    U[3][0] = 1
    # V: B00 -> V[0*3+0]=V[0], B11 -> V[1*3+1]=V[4]
    V[0][0] = 1
    V[4][0] = 1
    # W: C00 += M1 -> W[0*2+0]=W[0], C11 += M1 -> W[1*2+1]=W[3]
    W[0][0] = 1
    W[3][0] = 1
    
    # Column 1: M2 = (A10+A11)B00
    # U: A10 -> U[1*2+0]=U[2], A11 -> U[3]
    U[2][1] = 1
    U[3][1] = 1
    # V: B00 -> V[0]
    V[0][1] = 1
    # W: C10 += M2 -> W[0*2+1]=W[1], C11 += M2 -> W[3] with -1
    W[1][1] = 1
    W[3][1] = -1
    
    # Column 2: M3 = A00(B01-B11)
    # U: A00 -> U[0]
    U[0][2] = 1
    # V: B01 -> V[0*3+1]=V[1], B11 -> V[4] with -1
    V[1][2] = 1
    V[4][2] = -1
    # W: C01 += M3 -> W[1*2+0]=W[2], C11 += M3 -> W[3]
    W[2][2] = 1
    W[3][2] = 1
    
    # Column 3: M4 = A11(B10-B00)
    # U: A11 -> U[3]
    U[3][3] = 1
    # V: B10 -> V[1*3+0]=V[3], B00 -> V[0] with -1
    V[3][3] = 1
    V[0][3] = -1
    # W: C00 += M4 -> W[0], C10 += M4 -> W[1]
    W[0][3] = 1
    W[1][3] = 1
    
    # Column 4: M5 = (A00+A10)B11
    # U: A00 -> U[0], A10 -> U[2]
    U[0][4] = 1
    U[2][4] = 1
    # V: B11 -> V[4]
    V[4][4] = 1
    # W: C00 -= M5 -> W[0] with -1, C01 += M5 -> W[2]
    W[0][4] = -1
    W[2][4] = 1
    
    # Column 5: M6 = (A01-A11)(B10+B11)
    # U: A01 -> U[0*2+1]=U[1], A11 -> U[3] with -1
    U[1][5] = 1
    U[3][5] = -1
    # V: B10 -> V[3], B11 -> V[4]
    V[3][5] = 1
    V[4][5] = 1
    # W: C11 += M6 -> W[3]
    W[3][5] = 1
    
    # Column 6: M7 = (A00-A10)(B00+B10)
    # U: A00 -> U[0], A10 -> U[2] with -1
    U[0][6] = 1
    U[2][6] = -1
    # V: B00 -> V[0], B10 -> V[3]
    V[0][6] = 1
    V[3][6] = 1
    # W: C00 += M7 -> W[0]
    W[0][6] = 1
    
    # Schoolbook part: columns 7-10 for the 2x2x1 subproblem (column 2 of B)
    # B[j,2] for j in {0,1}, C[i,2] for i in {0,1}
    # C[i,2] = sum_j A[i,j] * B[j,2]
    
    # Column 7: A00 * B02
    # U: A00 -> U[0]
    U[0][7] = 1
    # V: B02 -> V[0*3+2]=V[2]
    V[2][7] = 1
    # W: C02 += -> W[2*2+0]=W[4]
    W[4][7] = 1
    
    # Column 8: A01 * B02
    # U: A01 -> U[1]
    U[1][8] = 1
    # V: B02 -> V[2]
    V[2][8] = 1
    # W: C02 += -> W[4]
    W[4][8] = 1
    
    # Column 9: A10 * B12
    # U: A10 -> U[2]
    U[2][9] = 1
    # V: B12 -> V[1*3+2]=V[5]
    V[5][9] = 1
    # W: C12 += -> W[2*2+1]=W[5]
    W[5][9] = 1
    
    # Column 10: A11 * B12
    # U: A11 -> U[3]
    U[3][10] = 1
    # V: B12 -> V[5]
    V[5][10] = 1
    # W: C12 += -> W[5]
    W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
