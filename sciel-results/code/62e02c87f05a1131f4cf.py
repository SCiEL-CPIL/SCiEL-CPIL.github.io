import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Rank 54 decomposition
    rank = 54
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    # a_idx: list of 4 indices for A (i*m+j)
    # b_idx: list of 4 indices for B (j*p+k)
    # w_idx: list of 4 indices for W (k*n+i)
    # The block computes C = A_sub * B_sub
    def add_strassen_block(a_idx, b_idx, w_idx):
        nonlocal col
        # a_idx corresponds to:
        # a00, a01
        # a10, a11
        # b_idx corresponds to:
        # b00, b01
        # b10, b11
        # w_idx corresponds to:
        # w00, w01
        # w10, w11
        
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0]][col] += 1.0
        U[a_idx[3]][col] += 1.0
        V[b_idx[0]][col] += 1.0
        V[b_idx[3]][col] += 1.0
        W[w_idx[0]][col] += 1.0  # w00 += M1
        W[w_idx[1]][col] += 0.0
        W[w_idx[2]][col] += 0.0
        W[w_idx[3]][col] += 1.0  # w11 += M1
        col += 1
        
        # M2 = (a10+a11)b00
        U[a_idx[2]][col] += 1.0
        U[a_idx[3]][col] += 1.0
        V[b_idx[0]][col] += 1.0
        W[w_idx[2]][col] += 1.0  # w10 += M2
        W[w_idx[3]][col] += -1.0 # w11 -= M2
        col += 1
        
        # M3 = a00(b01-b11)
        U[a_idx[0]][col] += 1.0
        V[b_idx[1]][col] += 1.0
        V[b_idx[3]][col] += -1.0
        W[w_idx[1]][col] += 1.0  # w01 += M3
        W[w_idx[3]][col] += 1.0  # w11 += M3
        col += 1
        
        # M4 = a11(b10-b00)
        U[a_idx[3]][col] += 1.0
        V[b_idx[2]][col] += 1.0
        V[b_idx[0]][col] += -1.0
        W[w_idx[0]][col] += 1.0  # w00 += M4
        W[w_idx[2]][col] += 1.0  # w10 += M4
        col += 1
        
        # M5 = (a00+a01)b11
        U[a_idx[0]][col] += 1.0
        U[a_idx[1]][col] += 1.0
        V[b_idx[3]][col] += 1.0
        W[w_idx[0]][col] += -1.0 # w00 -= M5
        W[w_idx[1]][col] += 1.0  # w01 += M5
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a_idx[2]][col] += 1.0
        U[a_idx[0]][col] += -1.0
        V[b_idx[0]][col] += 1.0
        V[b_idx[1]][col] += 1.0
        W[w_idx[3]][col] += 1.0  # w11 += M6
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a_idx[1]][col] += 1.0
        U[a_idx[3]][col] += -1.0
        V[b_idx[2]][col] += 1.0
        V[b_idx[3]][col] += 1.0
        W[w_idx[0]][col] += 1.0  # w00 += M7
        col += 1

    # Define indices for the 6 Strassen blocks
    # A is 2x5. Indices: i*5+j
    # A_L: i in {0,1}, j in {0,1} -> [0, 1, 5, 6]
    # A_R: i in {0,1}, j in {2,3} -> [2, 3, 7, 8]
    
    # B is 5x6. Indices: j*6+k
    # B_11: j in {0,1}, k in {0,1} -> [0, 1, 6, 7]
    # B_12: j in {0,1}, k in {2,3} -> [2, 3, 8, 9]
    # B_13: j in {0,1}, k in {4,5} -> [4, 5, 10, 11]
    # B_21: j in {2,3}, k in {0,1} -> [12, 13, 18, 19]
    # B_22: j in {2,3}, k in {2,3} -> [14, 15, 20, 21]
    # B_23: j in {2,3}, k in {4,5} -> [16, 17, 22, 23]
    
    # W is 6x2 (p*n). Indices: k*2+i
    # C_11: i in {0,1}, k in {0,1} -> [0, 1, 2, 3]
    # C_12: i in {0,1}, k in {2,3} -> [4, 5, 6, 7]
    # C_13: i in {0,1}, k in {4,5} -> [8, 9, 10, 11]
    # C_21: i in {0,1}, k in {0,1} -> [0, 1, 2, 3]
    # C_22: i in {0,1}, k in {2,3} -> [4, 5, 6, 7]
    # C_23: i in {0,1}, k in {4,5} -> [8, 9, 10, 11]
    
    a_L = [0, 1, 5, 6]
    a_R = [2, 3, 7, 8]
    
    b_11 = [0, 1, 6, 7]
    b_12 = [2, 3, 8, 9]
    b_13 = [4, 5, 10, 11]
    b_21 = [12, 13, 18, 19]
    b_22 = [14, 15, 20, 21]
    b_23 = [16, 17, 22, 23]
    
    w_11 = [0, 1, 2, 3] # k=0,1
    w_12 = [4, 5, 6, 7] # k=2,3
    w_13 = [8, 9, 10, 11] # k=4,5
    
    # Block 1: A_L * B_11 -> C_11
    add_strassen_block(a_L, b_11, w_11)
    
    # Block 2: A_L * B_12 -> C_12
    add_strassen_block(a_L, b_12, w_12)
    
    # Block 3: A_L * B_13 -> C_13
    add_strassen_block(a_L, b_13, w_13)
    
    # Block 4: A_R * B_21 -> C_21 (which is also C_11 position? No, C_21 refers to the contribution from A_R)
    # Wait, C_21 in my naming above was for A_R * B_21. The output goes to the same C matrix positions.
    # C_11 output indices are [0,1,2,3].
    # C_21 output indices are [0,1,2,3].
    # So we add to the same W indices.
    add_strassen_block(a_R, b_21, w_11)
    
    # Block 5: A_R * B_22 -> C_22 (output to C_12 positions)
    add_strassen_block(a_R, b_22, w_12)
    
    # Block 6: A_R * B_23 -> C_23 (output to C_13 positions)
    add_strassen_block(a_R, b_23, w_13)
    
    # Tail: A_2 * B_2
    # A_2: j=4. Indices: 0*5+4=4, 1*5+4=9
    # B_2: j=4. Indices: 4*6+k = 24+k
    # Output: C[i,k] = A[i,4]*B[4,k]
    for k in range(6):
        # Term for i=0
        U[4][col] = 1.0
        V[24 + k][col] = 1.0
        W[k * 2 + 0][col] = 1.0
        col += 1
        
        # Term for i=1
        U[9][col] = 1.0
        V[24 + k][col] = 1.0
        W[k * 2 + 1][col] = 1.0
        col += 1
        
    if col != rank:
        # If col is not 54, something went wrong with the block counting
        # 6 blocks * 7 = 42. 12 tail terms. Total 54.
        pass

    return {"U": U, "V": V, "W": W}
