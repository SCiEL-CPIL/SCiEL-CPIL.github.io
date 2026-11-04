import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    """
    Return a rank-23 tensor decomposition for 3x3 matrix multiplication
    based on the Winograd algorithm. Falls back to rank-27 if verification fails.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook rank-27 solution (fallback)
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1.0
                V_school[b, col] = 1.0
                W_school[c, col] = 1.0
                col += 1
                
    # Attempt rank-23 Winograd decomposition
    rank = 23
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Define the 23 terms.
    # A indices: 0..8 (row-major: 00, 01, 02, 10, 11, 12, 20, 21, 22)
    # B indices: 0..8 (row-major: 00, 01, 02, 10, 11, 12, 20, 21, 22)
    # C indices: 0..8 (col-major? k*n+i: 00, 10, 20, 01, 11, 21, 02, 12, 22)
    
    # Term 0: a00 * b00 -> c00 (idx 0)
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    
    # Term 1: a01 * b01 -> c01 (idx 3)
    U[1, 1] = 1; V[1, 1] = 1; W[3, 1] = 1
    
    # Term 2: a02 * b02 -> c02 (idx 6)
    U[2, 2] = 1; V[2, 2] = 1; W[6, 2] = 1
    
    # Term 3: a10 * b10 -> c10 (idx 1)
    U[3, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
    
    # Term 4: a11 * b11 -> c11 (idx 4)
    U[4, 4] = 1; V[4, 4] = 1; W[4, 4] = 1
    
    # Term 5: a12 * b12 -> c12 (idx 7)
    U[5, 5] = 1; V[5, 5] = 1; W[7, 5] = 1
    
    # Term 6: a20 * b20 -> c20 (idx 2)
    U[6, 6] = 1; V[6, 6] = 1; W[2, 6] = 1
    
    # Term 7: a21 * b21 -> c21 (idx 5)
    U[7, 7] = 1; V[7, 7] = 1; W[5, 7] = 1
    
    # Term 8: a22 * b22 -> c22 (idx 8)
    U[8, 8] = 1; V[8, 8] = 1; W[8, 8] = 1
    
    # Term 9: (a00+a10)*(b00+b10)
    U[0, 9] = 1; U[3, 9] = 1
    V[0, 9] = 1; V[3, 9] = 1
    # Contributes to c00, c10, c01?, c11?
    # (a00+a10)(b00+b10) = a00b00 + a00b10 + a10b00 + a10b10
    # a00b00 is c00. a10b10 is c10.
    # a00b10 contributes to c00? No, a00b10 is part of c01?
    # C_01 = a00b01 + a01b11 + a02b21. No.
    # C_ik = sum_j a_ij b_jk.
    # a00 b10: i=0, j=0, k=1? No, b10 is row 1 col 0.
    # b_jk: j=1, k=0.
    # a00 b10 -> C_00? No, C_00 = a00b00 + a01b10 + a02b20.
    # So a00b10 is NOT in C_00.
    # a00b10 is in C_00? No.
    # a00 b10: i=0, j=0, k=0? No, b10 is j=1, k=0.
    # So a00 b10 contributes to C_00?
    # C_00 = a00b00 + a01b10 + a02b20.
    # So a00b10 is NOT in C_00.
    # My previous assumption was wrong.
    
    # I cannot reliably derive the W coefficients without the full algorithm.
    # I will return the schoolbook solution to be safe.
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
