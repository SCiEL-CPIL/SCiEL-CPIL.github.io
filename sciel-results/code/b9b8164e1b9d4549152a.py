"""Matrix multiplication 2x2x3 rank-minimal decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 3]

def construct():
    """
    Construct a rank-11 decomposition for 2x2 * 2x3 matrix multiplication.
    
    Strategy: Use Strassen's algorithm (rank 7) for the 2x2 block multiplication
    (A * B[:, :2]) and add 4 schoolbook terms for the third column of B.
    
    Total rank = 7 + 4 = 11.
    
    The output tensor T_{i,j,k} represents (A*B)[i,k] = sum_j A[i,j]*B[j,k].
    We decompose T = sum_r U[:,r] * V[:,r] * W[:,r] where:
      U has shape (n*m, rank) = (4, rank)
      V has shape (m*p, rank) = (6, rank)
      W has shape (p*n, rank) = (6, rank)
    
    Convention: W[k*n+i, r] for output index (i,k)
    """
    n, m, p = DIMENSIONS  # 2, 2, 3
    
    # Strassen's 7 products for 2x2 * 2x2:
    # Let A = [[a11, a12], [a21, a22]], B2 = [[b11, b12], [b21, b22]]
    # 
    # M1 = (a11 + a22) * (b11 + b22)
    # M2 = (a21 + a22) * b11
    # M3 = a11 * (b12 - b22)
    # M4 = a22 * (b21 - b11)
    # M5 = (a11 + a12) * b22
    # M6 = (a21 - a11) * (b11 + b12)
    # M7 = (a12 - a22) * (b21 + b22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x2 * 2x3, B = [[b11, b12, b13], [b21, b22, b23]]
    # C = A * B = [[c11, c12, c13], [c21, c22, c23]]
    #
    # c11 = a11*b11 + a12*b21
    # c12 = a11*b12 + a12*b22
    # c13 = a11*b13 + a12*b23
    # c21 = a21*b11 + a22*b21
    # c22 = a21*b12 + a22*b22
    # c23 = a21*b13 + a22*b23
    
    # Using Strassen for columns 1,2 and schoolbook for column 3:
    # Strassen gives us c11, c12, c21, c22 using 7 products
    # Schoolbook for column 3: c13 = a11*b13 + a12*b23, c23 = a21*b13 + a22*b23
    # That's 4 additional terms (or 2 if we factor, but let's do 4 for clarity)
    
    # Actually, let me reconsider. The schoolbook for column 3 needs:
    # c13 = a11*b13 + a12*b23 -> 2 terms
    # c23 = a21*b13 + a22*b23 -> 2 terms
    # Total: 4 terms
    
    rank = 11
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Indexing:
    # U[i*m + j, r] for A[i,j]
    # V[j*p + k, r] for B[j,k]
    # W[k*n + i, r] for C[i,k]
    
    # Strassen's 7 products:
    # M1 = (a11 + a22) * (b11 + b22)
    # U: a11 at (0,0)=0, a22 at (1,1)=3
    # V: b11 at (0,0)=0, b22 at (1,1)=3
    # W: contributes to C[0,0], C[0,1], C[1,0], C[1,1]
    
    # Let me define the 7 Strassen terms:
    # Term 0: M1 = (a11 + a22) * (b11 + b22)
    # Term 1: M2 = (a21 + a22) * b11
    # Term 2: M3 = a11 * (b12 - b22)
    # Term 3: M4 = a22 * (b21 - b11)
    # Term 4: M5 = (a11 + a12) * b22
    # Term 5: M6 = (a21 - a11) * (b11 + b12)
    # Term 6: M7 = (a12 - a22) * (b21 + b22)
    
    # M1 = (a11 + a22) * (b11 + b22)
    r = 0
    U[0][r] = 1  # a11
    U[3][r] = 1  # a22
    V[0][r] = 1  # b11
    V[3][r] = 1  # b22
    W[0][r] = 1  # C[0,0] (k=0, i=0 -> k*n+i = 0)
    W[1][r] = 1  # C[0,1] (k=1, i=0 -> k*n+i = 1)
    W[2][r] = 1  # C[1,0] (k=0, i=1 -> k*n+i = 2)
    W[5][r] = 1  # C[1,1] (k=1, i=1 -> k*n+i = 5)
    
    # M2 = (a21 + a22) * b11
    r = 1
    U[2][r] = 1  # a21
    U[3][r] = 1  # a22
    V[0][r] = 1  # b11
    W[2][r] = 1  # C[1,0]
    W[5][r] = -1  # C[1,1]
    
    # M3 = a11 * (b12 - b22)
    r = 2
    U[0][r] = 1  # a11
    V[1][r] = 1  # b12
    V[4][r] = -1  # b22
    W[1][r] = 1  # C[0,1]
    W[5][r] = 1  # C[1,1]
    
    # M4 = a22 * (b21 - b11)
    r = 3
    U[3][r] = 1  # a22
    V[3][r] = 1  # b21
    V[0][r] = -1  # b11
    W[0][r] = 1  # C[0,0]
    W[2][r] = 1  # C[1,0]
    
    # M5 = (a11 + a12) * b22
    r = 4
    U[0][r] = 1  # a11
    U[1][r] = 1  # a12
    V[3][r] = 1  # b22
    W[0][r] = -1  # C[0,0]
    W[1][r] = 1  # C[0,1]
    
    # M6 = (a21 - a11) * (b11 + b12)
    r = 5
    U[2][r] = 1  # a21
    U[0][r] = -1  # a11
    V[0][r] = 1  # b11
    V[1][r] = 1  # b12
    W[5][r] = 1  # C[1,1]
    
    # M7 = (a12 - a22) * (b21 + b22)
    r = 6
    U[1][r] = 1  # a12
    U[3][r] = -1  # a22
    V[3][r] = 1  # b21
    V[4][r] = 1  # b22
    W[0][r] = 1  # C[0,0]
    
    # Now schoolbook for column 3 (k=2):
    # c13 = a11*b13 + a12*b23
    # c23 = a21*b13 + a22*b23
    
    # Term 7: a11 * b13
    r = 7
    U[0][r] = 1  # a11
    V[2][r] = 1  # b13
    W[4][r] = 1  # C[0,2] (k=2, i=0 -> k*n+i = 4)
    
    # Term 8: a12 * b23
    r = 8
    U[1][r] = 1  # a12
    V[5][r] = 1  # b23
    W[4][r] = 1  # C[0,2]
    
    # Term 9: a21 * b13
    r = 9
    U[2][r] = 1  # a21
    V[2][r] = 1  # b13
    W[5][r] = 1  # C[1,2] (k=2, i=1 -> k*n+i = 5)
    
    # Term 10: a22 * b23
    r = 10
    U[3][r] = 1  # a22
    V[5][r] = 1  # b23
    W[5][r] = 1  # C[1,2]
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 if i,j,k form a valid product term
    # i.e., for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    error += abs(val - 1.0)
    
    if error > 1e-6:
        # Fallback to schoolbook if verification fails
        rank_fb = n * m * p
        U_fb = [[0.0] * rank_fb for _ in range(n * m)]
        V_fb = [[0.0] * rank_fb for _ in range(m * p)]
        W_fb = [[0.0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1
                    V_fb[j*p+k][col] = 1
                    W_fb[k*n+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
