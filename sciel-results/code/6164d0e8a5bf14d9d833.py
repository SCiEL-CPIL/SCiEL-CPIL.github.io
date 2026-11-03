import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

def verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    S = np.einsum('ijt,jkt,kit->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if not np.allclose(S, 1.0, atol=1e-9):
        return False
    return True

def _strassen_2x2x2():
    """
    Strassen's rank-7 algorithm for 2x2x2.
    Returns U, V, W arrays.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen's 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U[i*m+j, k] = coefficient of A[i,j] in product k
    # V[j*p+k, k] = coefficient of B[j,k] in product k
    # W[k*n+i, k] = coefficient for C[i,k] in product k
    
    # Product 0 (M1): (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1; W[3, 0] = 1  # C00, C11
    
    # Product 1 (M2): (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1; W[3, 1] = -1  # C10, -C11
    
    # Product 2 (M3): A00*(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1; V[3, 2] = -1  # B01, -B11
    W[1, 2] = 1; W[3, 2] = 1  # C01, C11
    
    # Product 3 (M4): A11*(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1; V[0, 3] = -1  # B10, -B00
    W[0, 3] = 1; W[2, 3] = 1  # C00, C10
    
    # Product 4 (M5): (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1; W[1, 4] = 1  # -C00, C01
    
    # Product 5 (M6): (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1  # A10, -A00
    V[0, 5] = 1; V[1, 5] = 1  # B00, B01
    W[3, 5] = 1  # C11
    
    # Product 6 (M7): (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1  # A01, -A11
    V[2, 6] = 1; V[3, 6] = 1  # B10, B11
    W[0, 6] = 1  # C00
    
    return U, V, W

def _strassen_4x4x4():
    """
    Recursive Strassen for 4x4x4 using 2x2x2 blocks.
    Rank = 7 * 7 = 49.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U1, V1, W1 = _strassen_2x2x2()
    
    # Kronecker product of Strassen 2x2x2 with itself
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    r1 = 7
    for l in range(r1):
        for k in range(r1):
            idx = l * r1 + k
            # U1_3d[i1, j1, l], U2_3d[i2, j2, k]
            U1_3d = U1.reshape(2, 2, r1)
            V1_3d = V1.reshape(2, 2, r1)
            W1_3d = W1.reshape(2, 2, r1)
            
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            U[i1*2+i2, j1*2+j2, idx] = U1_3d[i1, j1, l] * U1_3d[i2, j2, k]
            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            V[j1*2+j2, k1*2+k2, idx] = V1_3d[j1, k1, l] * V1_3d[j2, k2, k]
            
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            W[k1*2+k2, i1*2+i2, idx] = W1_3d[k1, i1, l] * W1_3d[k2, i2, k]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Try Strassen 4x4x4 (rank 49) extended to 4x4x9
    # 4x4x9: split into 4x4x4 + 4x4x4 + 4x4x1
    # Use rank 49 for first two, rank 4 for last
    # Total rank: 49 + 49 + 4 = 102
    
    U_strassen, V_strassen, W_strassen = _strassen_4x4x4()
    
    # Schoolbook for 4x4x1
    U_sb1 = np.zeros((n*m, 4))
    V_sb1 = np.zeros((m*1, 4))
    W_sb1 = np.zeros((1*n, 4))
    col = 0
    for i in range(n):
        for j in range(m):
            U_sb1[i*m+j, col] = 1
            V_sb1[j, col] = 1
            W_sb1[i, col] = 1
            col += 1
    
    # Combine: 4x4x9 = [4x4x4 | 4x4x4 | 4x4x1]
    # U: (16, 49+49+4) = (16, 102)
    # V: (4*9, 102) = (36, 102)
    # W: (9*4, 102) = (36, 102)
    
    total_rank = 49 + 49 + 4
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # First 4x4x4 block (columns 0-3 of B)
    U[:, 0:49] = U_strassen
    V[:, 0:49] = V_strassen  # V is (16, 49), but we need (36, 102)
    # V_strassen is for 4x4x4, so V_strassen has shape (4*4, 49) = (16, 49)
    # For 4x4x9, V should be (4*9, 102) = (36, 102)
    # The first 4 columns of B correspond to V rows 0-15 (j=0..3, k=0..3)
    V[0:16, 0:49] = V_strassen
    # W_strassen is (4*4, 49) = (16, 49), but W should be (9*4, 102) = (36, 102)
    # W is indexed [k*n+i, rank], so for k=0..3, i=0..3: rows 0..15
    W[0:16, 0:49] = W_strassen
    
    # Second 4x4x4 block (columns 4-7 of B)
    U[:, 49:98] = U_strassen
    V[16:32, 49:98] = V_strassen  # j=0..3, k=4..7 → rows 16..31
    W[16:32, 49:98] = W_strassen  # k=4..7, i=0..3 → rows 16..31
    
    # Third 4x4x1 block (column 8 of B)
    U[:, 98:102] = U_sb1
    V[32:36, 98:102] = V_sb1  # j=0..3, k=8 → rows 32..35
    W[32:36, 98:102] = W_sb1  # k=8, i=0..3 → rows 32..35
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
