"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2: rank 7
    # U[i,j,c], V[j,k,c], W[k,i,c] for i,j,k in {0,1}, c in {0..6}
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A10-A11)(B01+B11)
    # C00 = M1+M4-M5-M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M7
    
    # U for 2x2x2 Strassen (rows: i*m+j = i*2+j, so 00,01,10,11)
    # V for 2x2x2 Strassen (rows: j*p+k = j*2+k, so 00,01,10,11)
    # W for 2x2x2 Strassen (rows: k*n+i = k*2+i, so 00,01,10,11)
    
    # U[i*m+j, c] for i,j in {0,1}, c in {0..6}
    # Rows: 00, 01, 10, 11
    U_222 = [
        [1, 0, 1, 0, 1, -1, 0],  # row 00 (A[0,0])
        [0, 0, 0, 0, 0, 1, 0],   # row 01 (A[0,1])
        [0, 1, 0, 0, 1, 0, 1],   # row 10 (A[1,0])
        [1, 1, 0, 1, 0, 0, -1],  # row 11 (A[1,1])
    ]
    
    # V[j*p+k, c] for j,k in {0,1}, c in {0..6}
    # Rows: 00, 01, 10, 11
    V_222 = [
        [1, 1, 0, 0, 0, 1, 0],  # row 00 (B[0,0])
        [1, 0, 0, 1, 0, 0, 1],  # row 01 (B[0,1])
        [1, 0, 1, -1, 0, 1, 0], # row 10 (B[1,0])
        [1, 0, -1, 0, 1, 0, 1], # row 11 (B[1,1])
    ]
    
    # W[k*n+i, c] for k,i in {0,1}, c in {0..6}
    # Rows: 00, 01, 10, 11 (k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1)
    W_222 = [
        [1, 0, 0, 1, -1, 0, -1], # row 0: C[0,0]
        [0, 1, 0, 1, 0, 0, 0],   # row 1: C[1,0]
        [0, 0, 1, 0, 1, 0, 0],   # row 2: C[0,1]
        [1, -1, 1, 0, 0, 0, 1],  # row 3: C[1,1]
    ]
    
    # Verify Strassen 2x2x2
    def verify_222():
        U_r = np.array(U_222, dtype=float).reshape(4, 7)
        V_r = np.array(V_222, dtype=float).reshape(4, 7)
        W_r = np.array(W_222, dtype=float).reshape(4, 7)
        T = np.einsum('ijc,jkc,kic->ijk', U_r.reshape(2,2,7), V_r.reshape(2,2,7), W_r.reshape(2,2,7))
        return np.allclose(T, np.ones((2,2,2)), atol=1e-9)
    
    # Schoolbook for other block sizes
    def schoolbook_block(a, b, c):
        """Return U, V, W for a x b x c schoolbook decomposition."""
        rank = a * b * c
        U = [[0]*rank for _ in range(a*b)]
        V = [[0]*rank for _ in range(b*c)]
        W = [[0]*rank for _ in range(c*a)]
        col = 0
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    U[i*b+j][col] = 1
                    V[j*c+k][col] = 1
                    W[k*a+i][col] = 1
                    col += 1
        return U, V, W
    
    # Use schoolbook decomposition as the safe, correct answer
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, column] = 1
                V_school[j * p + k, column] = 1
                W_school[k * n + i, column] = 1
                column += 1
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
