"""Rewritten program for 2x2 matrix multiplication rank minimization."""
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x2 matrix multiplication.
    
    The tensor rank of 2x2 matrix multiplication is known to be 7 (Strassen's algorithm).
    No rank-6 decomposition exists over any field. We return the verified rank-7 Strassen
    decomposition with integer entries.
    
    Interface:
    - U: shape (n*m, rank) = (4, 7), U[i*m + a, r]
    - V: shape (m*p, rank) = (4, 7), V[a*p + k, r]  
    - W: shape (p*n, rank) = (4, 7), W[k*n + i, r]
    
    The decomposition satisfies:
    (A*B)[i,k] = sum_{r=0}^{6} W[k*n+i, r] * (sum_{a} U[i*m+a, r] * A[i,a]) * (sum_{k'} V[a*p+k', r] * B[a,k'])
    
    More precisely, each rank-r term computes a product of a linear combination of A entries
    times a linear combination of B entries, and W gives the coefficient for that product
    in each output entry.
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # M1: A part: A00 + A11, B part: B00 + B11
    U[0][0] = 1   # A[0,0]
    U[3][0] = 1   # A[1,1]
    V[0][0] = 1   # B[0,0]
    V[3][0] = 1   # B[1,1]
    
    # M2: A part: A10 + A11, B part: B00
    U[2][1] = 1   # A[1,0]
    U[3][1] = 1   # A[1,1]
    V[0][1] = 1   # B[0,0]
    
    # M3: A part: A00, B part: B01 - B11
    U[0][2] = 1   # A[0,0]
    V[1][2] = 1   # B[0,1]
    V[3][2] = -1  # -B[1,1]
    
    # M4: A part: A11, B part: B10 - B00
    U[3][3] = 1   # A[1,1]
    V[2][3] = 1   # B[1,0]
    V[0][3] = -1  # -B[0,0]
    
    # M5: A part: A00 + A01, B part: B11
    U[0][4] = 1   # A[0,0]
    U[1][4] = 1   # A[0,1]
    V[3][4] = 1   # B[1,1]
    
    # M6: A part: A10 - A00, B part: B00 + B01
    U[2][5] = 1   # A[1,0]
    U[0][5] = -1  # -A[0,0]
    V[0][5] = 1   # B[0,0]
    V[1][5] = 1   # B[0,1]
    
    # M7: A part: A01 - A11, B part: B10 + B11
    U[1][6] = 1   # A[0,1]
    U[3][6] = -1  # -A[1,1]
    V[2][6] = 1   # B[1,0]
    V[3][6] = 1   # B[1,1]
    
    # W[k*n + i, r] where k is column of C, i is row of C
    # C[0,0]: k=0, i=0 -> W[0*2+0, r] = W[0, r]
    # C[0,1]: k=1, i=0 -> W[1*2+0, r] = W[2, r]
    # C[1,0]: k=0, i=1 -> W[0*2+1, r] = W[1, r]
    # C[1,1]: k=1, i=1 -> W[1*2+1, r] = W[3, r]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0][0] = 1   # M1
    W[0][3] = 1   # M4
    W[0][4] = -1  # -M5
    W[0][6] = 1   # M7
    
    # C[0,1] = M3 + M5
    W[2][2] = 1   # M3
    W[2][4] = 1   # M5
    
    # C[1,0] = M2 + M4
    W[1][1] = 1   # M2
    W[1][3] = 1   # M4
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3][0] = 1   # M1
    W[3][1] = -1  # -M2
    W[3][2] = 1   # M3
    W[3][5] = 1   # M6
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
