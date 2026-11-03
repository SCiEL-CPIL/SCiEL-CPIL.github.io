import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    total_time_limit = 110.0
    start_time = time.time()
    
    # Rank 36 Solution
    # 1 term for j=0
    # 5 * 7 terms for j=1,2 (Strassen)
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part A: j=0
    # U[i*m + 0, col] = 1
    # V[0*p + k, col] = 1
    # W[k*n + i, col] = 1
    for i in range(n):
        U[i * m + 0, col] = 1
    for k in range(p):
        V[0 * p + k, col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i, col] = 1
    col += 1
    
    # Part B: j=1,2
    # 5 blocks of 2 columns
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Global indices
        # A: i in {0,1}, j in {1,2}
        u_01 = 0 * m + 1
        u_02 = 0 * m + 2
        u_11 = 1 * m + 1
        u_12 = 1 * m + 2
        
        # B: j in {1,2}, k in {k0, k1}
        v_10 = 1 * p + k0
        v_11 = 1 * p + k1
        v_20 = 2 * p + k0
        v_21 = 2 * p + k1
        
        # C: i in {0,1}, k in {k0, k1}
        w_00 = k0 * n + 0
        w_01 = k0 * n + 1
        w_10 = k1 * n + 0
        w_11 = k1 * n + 1
        
        # Strassen Terms
        # M1 = (A01 + A11)(B10 + B20) -> Wait, standard Strassen mapping:
        # A_sub = [[A01, A02], [A11, A12]]
        # B_sub = [[B10, B11], [B20, B21]]
        # C_sub = A_sub @ B_sub
        
        # M1 = (A01 + A11) * (B10 + B20) ? No.
        # Standard:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11) B00
        # M3 = A00 (B01 - B11)
        # M4 = A11 (B10 - B00)
        # M5 = (A00 + A01) B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # Map:
        # A00 -> u_01
        # A01 -> u_02
        # A10 -> u_11
        # A11 -> u_12
        
        # B00 -> v_10
        # B01 -> v_11
        # B10 -> v_20
        # B11 -> v_21
        
        # C00 -> w_00
        # C01 -> w_01
        # C10 -> w_10
        # C11 -> w_11
        
        # Term 1
        U[u_01, col] += 1; U[u_12, col] += 1
        V[v_10, col] += 1; V[v_21, col] += 1
        W[w_00, col] += 1; W[w_11, col] += 1
        col += 1
        
        # Term 2
        U[u_11, col] += 1; U[u_12, col] += 1
        V[v_10, col] += 1
        W[w_10, col] += 1; W[w_11, col] += -1
        col += 1
        
        # Term 3
        U[u_01, col] += 1
        V[v_11, col] += 1; V[v_21, col] += -1
        W[w_01, col] += 1; W[w_11, col] += 1
        col += 1
        
        # Term 4
        U[u_12, col] += 1
        V[v_20, col] += 1; V[v_10, col] += -1
        W[w_00, col] += 1; W[w_10, col] += 1
        col += 1
        
        # Term 5
        U[u_01, col] += 1; U[u_02, col] += 1
        V[v_21, col] += 1
        W[w_00, col] += -1; W[w_01, col] += 1
        col += 1
        
        # Term 6
        U[u_11, col] += 1; U[u_01, col] += -1
        V[v_10, col] += 1; V[v_11, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # Term 7
        U[u_02, col] += 1; U[u_12, col] += -1
        V[v_20, col] += 1; V[v_21, col] += 1
        W[w_00, col] += 1
        col += 1

    def to_list(mat):
        return mat.tolist()

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
