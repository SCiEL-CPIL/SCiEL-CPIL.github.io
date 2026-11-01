"""Matrix multiplication 2x2x2: Strassen rank-7 decomposition."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the Strassen rank-7 decomposition for 2x2 matrix multiplication.
    
    For A (2x2) * B (2x2) = C (2x2):
      A = [[a11, a12], [a21, a22]]
      B = [[b11, b12], [b21, b22]]
      C = [[c11, c12], [c21, c22]]
    
    Strassen's 7 multiplications:
      m1 = (a11 + a22) * (b11 + b22)
      m2 = (a21 + a22) * b11
      m3 = a11 * (b12 - b22)
      m4 = a22 * (b21 - b11)
      m5 = (a11 + a12) * b22
      m6 = (a21 - a11) * (b11 + b12)
      m7 = (a12 - a22) * (b21 + b22)
    
    Then:
      c11 = m1 + m4 - m5 + m7
      c12 = m3 + m5
      c21 = m2 + m4
      c22 = m1 - m2 + m3 + m6
    
    The tensor decomposition has:
      U: (n*m, rank) = (4, 7)  -- extracts linear combos from A
      V: (m*p, rank) = (4, 7)  -- extracts linear combos from B
      W: (p*n, rank) = (4, 7)  -- combines to get C entries
    
    Indexing:
      A is indexed as [i*m + j] where i is row, j is col (0-indexed)
        So A[0]=a11, A[1]=a12, A[2]=a21, A[3]=a22
      B is indexed as [j*p + k] where j is row, k is col
        So B[0]=b11, B[1]=b12, B[2]=b21, B[3]=b22
      C (output) is indexed as W[k*n + i, r] per the trace convention.
        Wait, the task says "W[k*n+i,r]" - let me re-read.
        
        "The output factor follows the official trace convention W[k*n+i,r]"
        
        This means W has shape (p*n, rank) = (4, 7), and W[row, col] where row = k*n + i.
        Here k indexes the column of B / column of C, and i indexes the row of A / row of C.
        So W[k*2 + i, r] contributes to C[i][k].
        
        C[0][0] = c11 -> k=0, i=0 -> W[0, r]
        C[0][1] = c12 -> k=1, i=0 -> W[1, r]
        C[1][0] = c21 -> k=0, i=1 -> W[2, r]
        C[1][1] = c22 -> k=1, i=1 -> W[3, r]
    
    U[i*m + j, r] = coefficient of A[i][j] in the r-th multiplication
    V[j*p + k, r] = coefficient of B[j][k] in the r-th multiplication
    W[k*n + i, r] = coefficient of the r-th product in C[i][k]
    
    Let me set up the 7 terms:
    
    r=0: m1 = (a11 + a22) * (b11 + b22)
      U: a11=1, a22=1 -> U[0,0]=1, U[3,0]=1, others 0
      V: b11=1, b22=1 -> V[0,0]=1, V[3,0]=1, others 0
      W: contributes to c11 (+1), c22 (+1) -> W[0,0]=1 (c11: k=0,i=0), W[3,0]=1 (c22: k=1,i=1)
    
    r=1: m2 = (a21 + a22) * b11
      U: a21=1, a22=1 -> U[2,1]=1, U[3,1]=1
      V: b11=1 -> V[0,1]=1
      W: contributes to c21 (+1) -> W[2,1]=1 (c21: k=0,i=1)
    
    r=2: m3 = a11 * (b12 - b22)
      U: a11=1 -> U[0,2]=1
      V: b12=1, b22=-1 -> V[1,2]=1, V[3,2]=-1
      W: contributes to c12 (+1), c22 (+1) -> W[1,2]=1 (c12: k=1,i=0), W[3,2]=1 (c22: k=1,i=1)
    
    r=3: m4 = a22 * (b21 - b11)
      U: a22=1 -> U[3,3]=1
      V: b21=1, b11=-1 -> V[2,3]=1, V[0,3]=-1
      W: contributes to c11 (+1), c21 (+1) -> W[0,3]=1 (c11), W[2,3]=1 (c21)
    
    r=4: m5 = (a11 + a12) * b22
      U: a11=1, a12=1 -> U[0,4]=1, U[1,4]=1
      V: b22=1 -> V[3,4]=1
      W: contributes to c11 (-1), c12 (+1) -> W[0,4]=-1, W[1,4]=1
    
    r=5: m6 = (a21 - a11) * (b11 + b12)
      U: a21=1, a11=-1 -> U[2,5]=1, U[0,5]=-1
      V: b11=1, b12=1 -> V[0,5]=1, V[1,5]=1
      W: contributes to c22 (+1) -> W[3,5]=1
    
    r=6: m7 = (a12 - a22) * (b21 + b22)
      U: a12=1, a22=-1 -> U[1,6]=1, U[3,6]=-1
      V: b21=1, b22=1 -> V[2,6]=1, V[3,6]=1
      W: contributes to c11 (+1) -> W[0,6]=1
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # r=0: m1 = (a11 + a22) * (b11 + b22)
    U[0][0] = 1   # a11
    U[3][0] = 1   # a22
    V[0][0] = 1   # b11
    V[3][0] = 1   # b22
    W[0][0] = 1   # c11
    W[3][0] = 1   # c22
    
    # r=1: m2 = (a21 + a22) * b11
    U[2][1] = 1   # a21
    U[3][1] = 1   # a22
    V[0][1] = 1   # b11
    W[2][1] = 1   # c21
    
    # r=2: m3 = a11 * (b12 - b22)
    U[0][2] = 1   # a11
    V[1][2] = 1   # b12
    V[3][2] = -1  # b22
    W[1][2] = 1   # c12
    W[3][2] = 1   # c22
    
    # r=3: m4 = a22 * (b21 - b11)
    U[3][3] = 1   # a22
    V[2][3] = 1   # b21
    V[0][3] = -1  # b11
    W[0][3] = 1   # c11
    W[2][3] = 1   # c21
    
    # r=4: m5 = (a11 + a12) * b22
    U[0][4] = 1   # a11
    U[1][4] = 1   # a12
    V[3][4] = 1   # b22
    W[0][4] = -1  # c11
    W[1][4] = 1   # c12
    
    # r=5: m6 = (a21 - a11) * (b11 + b12)
    U[2][5] = 1   # a21
    U[0][5] = -1  # a11
    V[0][5] = 1   # b11
    V[1][5] = 1   # b12
    W[3][5] = 1   # c22
    
    # r=6: m7 = (a12 - a22) * (b21 + b22)
    U[1][6] = 1   # a12
    U[3][6] = -1  # a22
    V[2][6] = 1   # b21
    V[3][6] = 1   # b22
    W[0][6] = 1   # c11
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
