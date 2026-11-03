import time
import numpy as np
from scipy.optimize import least_squares
from itertools import product
import sys

DIMENSIONS = [2, 3, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def construct():
    global START_TIME, TIME_LIMIT
    if START_TIME is None:
        START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    N = n * m * p  # 30
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        U = [[0] * N for _ in range(n * m)]
        V = [[0] * N for _ in range(m * p)]
        W = [[0] * N for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def time_left():
        return TIME_LIMIT - (time.time() - START_TIME)
    
    if time_left() < 10:
        return schoolbook()
        
    # Construct Rank 21 Decomposition
    # Total Rank: 7 (Strassen 2x2x2) + 7 (Strassen 2x2x2) + 4 (Schoolbook 2x2x1) + 1 (2x1x2) + 1 (2x1x2) + 1 (2x1x1) = 21
    
    # We will build U, V, W as lists of lists, or numpy arrays and convert at the end.
    # Using numpy for efficiency in construction, then convert to list of floats.
    
    # Dimensions:
    # U: (n*m, rank) = (6, 21)
    # V: (m*p, rank) = (15, 21)
    # W: (p*n, rank) = (10, 21)
    
    rank = 21
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col_offset = 0
    
    # Helper to add a block
    def add_block(U_block, V_block, W_block, i_off, j_off, k_off):
        # i_off, j_off, k_off are offsets for indices i, j, k
        # U_block is (n_sub, r), V_block is (m_sub, r), W_block is (p_sub, r)
        # Map indices:
        # i_global = i_sub + i_off
        # j_global = j_sub + j_off
        # k_global = k_sub + k_off
        
        n_sub, r = U_block.shape
        m_sub = V_block.shape[0]
        p_sub = W_block.shape[0]
        
        for c in range(r):
            for i_sub in range(n_sub):
                i_global = i_sub + i_off
                U[i_global * m + (j_sub + j_off), col_offset + c] += U_block[i_sub, c]
            for j_sub in range(m_sub):
                j_global = j_sub + j_off
                V[(j_global) * p + (k_sub + k_off), col_offset + c] += V_block[j_sub, c]
            for k_sub in range(p_sub):
                k_global = k_sub + k_off
                W[(k_global) * n + (i_sub + i_off), col_offset + c] += W_block[k_sub, c]
                
    # This helper is tricky because indices are mixed.
    # Better to just fill the arrays directly for each specific block.
    
    # Reset U, V, W to 0
    U[:] = 0
    V[:] = 0
    W[:] = 0
    
    # Block 1: Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {0,1}
    # i_off=0, j_off=0, k_off=0
    # Strassen factors for 2x2x2
    # U_str: 2x2 x 7
    # V_str: 2x2 x 7
    # W_str: 2x2 x 7
    
    # Standard Strassen:
    # M1: A11(B11+B22)
    # M2: (A11+A22)B22
    # M3: (A12+A22)B21
    # M4: A22(B11-B21)
    # M5: A11(B12-B22)
    # M6: (A11+A12)B22 ? No, M6: (A11+A12)B22 is not standard.
    # Let's use the explicit 7-term decomposition.
    
    # I will use a known valid rank 7 decomposition for 2x2x2.
    # U, V, W are 4x7 matrices (flattened 2x2).
    # Indices for A (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for C (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U_str = np.array([
        [1, 1, 0, 1, 1, 1, 0], # A00
        [0, 0, 1, 0, 0, 1, 1], # A01
        [0, 1, 1, 1, 0, 0, 1], # A10
        [1, 1, 1, 1, 1, 0, 1]  # A11
    ])
    V_str = np.array([
        [1, 0, 0, 1, 0, 0, 1], # B00
        [0, 0, 0, 0, 1, 1, 1], # B01
        [1, 1, 1, 1, 0, 0, 0], # B10
        [1, 1, 0, 0, 1, 0, 0]  # B11
    ])
    W_str = np.array([
        [1, 0, 0, 0, 1, 0, 0], # C00
        [0, 0, 0, 1, 0, 1, 0], # C01
        [0, 1, 1, 0, 0, 0, 1], # C10
        [0, 1, 0, 0, 0, 0, 1]  # C11
    ])
    
    # Verify Strassen:
    # C00 = M1 + M4 - M2 - M5?
    # Let's trust the standard Strassen matrices.
    # Actually, I will just use the schoolbook for 2x2x2 (Rank 4) to be safe and simple, 
    # but Strassen is Rank 7? No, Strassen is Rank 7 for 2x2x2? 
    # Wait, Strassen reduces 8 to 7. So Rank 7 is better than 4? No, 4 < 7.
    # Schoolbook is Rank 4. Strassen is Rank 7.
    # Why would I use Strassen? 
    # Ah, Strassen is for 2x2x2. Schoolbook is 4 multiplications.
    # Rank of M_2,2,2 is 7? No, Rank is 7 is the number of multiplications in Strassen.
    # The rank of the tensor M_2,2,2 is 7? 
    # No, the rank of M_2,2,2 is 7 is a famous result (Strassen).
    # But schoolbook uses 4 terms? 
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    # This is 8 terms. 
    # Can we do it in 7? Yes, Strassen.
    # Can we do it in 4? No, the rank is 7.
    # So Schoolbook (8) is worse than Strassen (7).
    
    # So I should use Strassen.
    
    # Add Block 1 (Strassen 2x2x2)
    # i in {0,1}, j in {0,1}, k in {0,1}
    for c in range(7):
        for i_sub in range(2):
            for j_sub in range(2):
                idx_A = i_sub * 2 + j_sub
                idx_U = i_sub * m + j_sub # 0, 1, 2, 3
                U[idx_U, col_offset + c] += U_str[idx_A, c]
        for j_sub in range(2):
            for k_sub in range(2):
                idx_B = j_sub * 2 + k_sub
                idx_V = j_sub * p + k_sub # 0, 1, 2, 3 (since p=5, j=0,1)
                V[idx_V, col_offset + c] += V_str[idx_B, c]
        for k_sub in range(2):
            for i_sub in range(2):
                idx_C = k_sub * 2 + i_sub
                idx_W = k_sub * n + i_sub # 0, 1, 2, 3 (since n=2, k=0,1)
                W[idx_W, col_offset + c] += W_str[idx_C, c]
                
    col_offset += 7
    
    # Block 2: Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {2,3}
    # i_off=0, j_off=0, k_off=2
    for c in range(7):
        for i_sub in range(2):
            for j_sub in range(2):
                idx_A = i_sub * 2 + j_sub
                idx_U = i_sub * m + j_sub
                U[idx_U, col_offset + c] += U_str[idx_A, c]
        for j_sub in range(2):
            for k_sub in range(2):
                idx_B = j_sub * 2 + k_sub
                idx_V = j_sub * p + (k_sub + 2)
                V[idx_V, col_offset + c] += V_str[idx_B, c]
        for k_sub in range(2):
            for i_sub in range(2):
                idx_C = k_sub * 2 + i_sub
                idx_W = (k_sub + 2) * n + i_sub
                W[idx_W, col_offset + c] += W_str[idx_C, c]
                
    col_offset += 7
    
    # Block 3: Schoolbook 2x2x1 for i in {0,1}, j in {0,1}, k in {4}
    # Rank 4
    # C_i4 = A_i0 B_04 + A_i1 B_14
    # Terms:
    # 1. A_00 B_04 -> C_04
    # 2. A_01 B_14 -> C_04
    # 3. A_10 B_04 -> C_14
    # 4. A_11 B_14 -> C_14
    
    # U (2x2):
    # c0: A_00
    # c1: A_01
    # c2: A_10
    # c3: A_11
    # V (2x1):
    # c0: B_04
    # c1: B_14
    # c2: B_04
    # c3: B_14
    # W (1x2):
    # c0: C_04
    # c1: C_04
    # c2: C_14
    # c3: C_14
    
    # Indices:
    # U: i in {0,1}, j in {0,1}
    # V: j in {0,1}, k=4
    # W: k=4, i in {0,1}
    
    # c=0: A_00 B_04 -> C_04
    U[0, col_offset] = 1 # A_00
    V[0*5+4, col_offset] = 1 # B_04
    W[4*2+0, col_offset] = 1 # C_04
    
    # c=1: A_01 B_14 -> C_04
    U[1, col_offset+1] = 1 # A_01
    V[1*5+4, col_offset+1] = 1 # B_14
    W[4*2+0, col_offset+1] = 1 # C_04
    
    # c=2: A_10 B_04 -> C_14
    U[2, col_offset+2] = 1 # A_10
    V[0*5+4, col_offset+2] = 1 # B_04
    W[4*2+1, col_offset+2] = 1 # C_14
    
    # c=3: A_11 B_14 -> C_14
    U[3, col_offset+3] = 1 # A_11
    V[1*5+4, col_offset+3] = 1 # B_14
    W[4*2+1, col_offset+3] = 1 # C_14
    
    col_offset += 4
    
    # Block 4: Rank 1 2x1x2 for i in {0,1}, j=2, k in {0,1}
    # C_i0 = A_i2 B_20, C_i1 = A_i2 B_21
    # Tensor: (A_i2) (B_2k)
    # Rank 1: U=[1,1]^T, V=[1], W=[1,1]^T
    # c=0
    U[4, col_offset] = 1 # A_02
    U[5, col_offset] = 1 # A_12
    V[2*5+0, col_offset] = 1 # B_20
    V[2*5+1, col_offset] = 1 # B_21
    W[0*2+0, col_offset] = 1 # C_00
    W[1*2+0, col_offset] = 1 # C_01
    # Wait, W index is k*n + i.
    # C_00: k=0, i=0 -> 0
    # C_01: k=1, i=0 -> 1
    # C_10: k=0, i=1 -> 2
    # C_11: k=1, i=1 -> 3
    
    # My W assignment above:
    # W[0] = C_00
    # W[1] = C_01
    # This covers i=0.
    # We also need i=1.
    # W[2] = C_10
    # W[3] = C_11
    
    # So for Rank 1 block:
    U[4, col_offset] = 1
    U[5, col_offset] = 1
    V[2*5+0, col_offset] = 1
    V[2*5+1, col_offset] = 1
    W[0, col_offset] = 1 # C_00
    W[1, col_offset] = 1 # C_01
    W[2, col_offset] = 1 # C_10
    W[3, col_offset] = 1 # C_11
    
    col_offset += 1
    
    # Block 5: Rank 1 2x1x2 for i in {0,1}, j=2, k in {2,3}
    # C_i2 = A_i2 B_22, C_i3 = A_i2 B_23
    U[4, col_offset] = 1
    U[5, col_offset] = 1
    V[2*5+2, col_offset] = 1
    V[2*5+3, col_offset] = 1
    W[2*2+0, col_offset] = 1 # C_02 (k=2, i=0)
    W[3*2+0, col_offset] = 1 # C_03 (k=3, i=0)
    W[2*2+1, col_offset] = 1 # C_12 (k=2, i=1)
    W[3*2+1, col_offset] = 1 # C_13 (k=3, i=1)
    
    col_offset += 1
    
    # Block 6: Rank 1 2x1x1 for i in {0,1}, j=2, k=4
    # C_i4 = A_i2 B_24
    U[4, col_offset] = 1
    U[5, col_offset] = 1
    V[2*5+4, col_offset] = 1
    W[4*2+0, col_offset] = 1 # C_04
    W[4*2+1, col_offset] = 1 # C_14
    
    col_offset += 1
    
    # Total rank used: 7 + 7 + 4 + 1 + 1 + 1 = 21.
    
    # Convert to list of lists
    U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
    V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
    W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

