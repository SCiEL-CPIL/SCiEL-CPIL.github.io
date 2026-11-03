import numpy as np
import itertools

DIMENSIONS = [2, 4, 8]

def _strassen_2x2():
    """
    Returns the 7 Strassen factors for 2x2 matrix multiplication.
    Each factor is a tuple (U, V, W) where U, V, W are 4-vectors (flattened 2x2).
    The tensor is T[a, b, c] = sum_t U[a,t] * V[b,t] * W[c,t]
    where a indexes A (2x2), b indexes B (2x2), c indexes C (2x2).
    
    Strassen's 7 products:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 - M4 + M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # A indices: A11=0, A12=1, A21=2, A22=3
    # B indices: B11=0, B12=1, B21=2, B22=3
    # C indices: C11=0, C12=1, C21=2, C22=3
    
    factors = []
    
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    U = [1, 0, 0, 1]
    V = [1, 0, 0, 1]
    W = [1, 0, 0, 1]
    factors.append((U, V, W))
    
    # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
    U = [0, 0, 1, 1]
    V = [1, 0, 0, 0]
    W = [0, 0, 1, -1]
    factors.append((U, V, W))
    
    # M3 = A11(B21-B22) -> C12 += M3, C22 += M3
    U = [1, 0, 0, 0]
    V = [0, 0, 1, -1]
    W = [0, 1, 0, 1]
    factors.append((U, V, W))
    
    # M4 = A22(B12-B11) -> C11 -= M4, C21 += M4
    U = [0, 0, 0, 1]
    V = [-1, 1, 0, 0]
    W = [-1, 0, 1, 0]
    factors.append((U, V, W))
    
    # M5 = (A11+A12)B22 -> C11 += M5, C12 += M5
    U = [1, 1, 0, 0]
    V = [0, 0, 0, 1]
    W = [1, 1, 0, 0]
    factors.append((U, V, W))
    
    # M6 = (A21-A11)(B11+B12) -> C22 += M6
    U = [-1, 0, 1, 0]
    V = [1, 1, 0, 0]
    W = [0, 0, 0, 1]
    factors.append((U, V, W))
    
    # M7 = (A12-A22)(B21+B22) -> C11 += M7
    U = [0, 1, 0, -1]
    V = [0, 0, 1, 1]
    W = [1, 0, 0, 0]
    factors.append((U, V, W))
    
    return factors

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # We decompose the 2x4x8 multiplication into blocks of 2x2x2.
    # A is 2x4, B is 4x8, C is 2x8.
    # We partition A's columns into 2 blocks of 2: A[:, 0:2], A[:, 2:4]
    # We partition B's rows into 2 blocks of 2, and columns into 4 blocks of 2.
    # We partition C's columns into 4 blocks of 2.
    
    # For each block pair (j, k) where j in {0,1} (row block of B / col block of A)
    # and k in {0,1,2,3} (col block of B / col block of C), we have a 2x2x2 multiplication.
    
    strassen_factors = _strassen_2x2()
    
    # Total rank: 8 blocks * 7 factors = 56
    total_rank = 8 * 7
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for j in range(2):  # row block of B, col block of A
        for k in range(4):  # col block of B, col block of C
            for (U_local, V_local, W_local) in strassen_factors:
                # Map local 2x2 indices to global indices
                # A: rows 0..1, cols 2j..2j+1
                # Global A index: i * m + (2j + a_col)
                for i in range(2):
                    for a_col in range(2):
                        global_a = i * m + (2 * j + a_col)
                        local_a = i * 2 + a_col
                        U[global_a, col] = U_local[local_a]
                
                # B: rows 2j..2j+1, cols 2k..2k+1
                # Global B index: (2j + b_row) * p + (2k + b_col)
                for b_row in range(2):
                    for b_col in range(2):
                        global_b = (2 * j + b_row) * p + (2 * k + b_col)
                        local_b = b_row * 2 + b_col
                        V[global_b, col] = V_local[local_b]
                
                # C: rows 0..1, cols 2k..2k+1
                # Global C index: i * p + (2k + c_col)
                for i in range(2):
                    for c_col in range(2):
                        global_c = i * p + (2 * k + c_col)
                        local_c = i * 2 + c_col
                        W[global_c, col] = W_local[local_c]
                
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
