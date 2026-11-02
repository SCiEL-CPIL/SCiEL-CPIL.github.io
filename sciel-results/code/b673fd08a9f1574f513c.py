import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # Strategy: Use a hybrid approach with Strassen on 2x2 blocks.
    # We split the middle dimension m=4 into two groups of size 2.
    # For each group, we have a 2x2 (A block) multiplied by 2x8 (B block).
    # We further split the B block (2x8) into four 2x2 blocks.
    # For each 2x2 * 2x2 multiplication, we use Strassen's algorithm (rank 7).
    # Total rank = 2 (A blocks) * 4 (B blocks) * 7 (Strassen) = 56.
    
    # Strassen decomposition for 2x2 matrix multiplication:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Local indices for 2x2 matrices:
    # A: [0, 1, 2, 3] = [A11, A12, A21, A22]
    # B: [0, 1, 2, 3] = [B11, B12, B21, B22]
    # C: [0, 1, 2, 3] = [C11, C12, C21, C22]
    
    # U_local: 4 rows (A indices), 7 cols (Strassen terms)
    # V_local: 4 rows (B indices), 7 cols (Strassen terms)
    # W_local: 4 rows (C indices), 7 cols (Strassen terms)
    
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # M1: A11+A22, B11+B22 -> C11, C22
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1
    W_local[3, 0] = 1
    
    # M2: A21+A22, B11 -> C21, C22
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1
    W_local[3, 1] = -1
    
    # M3: A11, B12-B22 -> C12, C22
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1
    W_local[3, 2] = 1
    
    # M4: A22, B21-B11 -> C11, C21
    U_local[3, 3] = 1
    V_local[2, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1
    W_local[2, 3] = 1
    
    # M5: A11+A12, B22 -> C11, C12
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1
    W_local[1, 4] = 1
    
    # M6: A21-A11, B11+B12 -> C22
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1
    
    # M7: A12-A22, B21+B22 -> C11
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # A is 2x4, indexed as A[i, j] with i in {0,1}, j in {0,1,2,3}
    # Flattened A index: i * m + j = i * 4 + j
    # B is 4x8, indexed as B[j, k] with j in {0,1,2,3}, k in {0,...,7}
    # Flattened B index: j * p + k = j * 8 + k
    # C is 2x8, indexed as C[i, k] with i in {0,1}, k in {0,...,7}
    # Flattened C index: k * n + i (trace convention W[k*n+i, r])
    
    # We split j (middle dim) into two groups: j_group=0 for j in {0,1}, j_group=1 for j in {2,3}
    # For each j_group, A_block is 2x2 (rows i=0,1; cols j=j_group*2, j_group*2+1)
    # For each j_group, B_block is 2x8 (rows j=j_group*2, j_group*2+1; cols k=0,...,7)
    
    for j_group in range(2):
        # A block: 2x2, local indices [0,1,2,3] correspond to global:
        # a_map[0] = i=0, j=j_group*2
        # a_map[1] = i=0, j=j_group*2+1
        # a_map[2] = i=1, j=j_group*2
        # a_map[3] = i=1, j=j_group*2+1
        a_map = [
            0 * 4 + j_group * 2,
            0 * 4 + j_group * 2 + 1,
            1 * 4 + j_group * 2,
            1 * 4 + j_group * 2 + 1
        ]
        
        # B block: 2x8, we split k into 4 groups of 2: k_group in {0,1,2,3}
        # For each k_group, B_sub is 2x2 (rows j=j_group*2, j_group*2+1; cols k=k_group*2, k_group*2+1)
        for k_group in range(4):
            # b_map for this 2x2 sub-block:
            # b_map[0] = j=j_group*2, k=k_group*2
            # b_map[1] = j=j_group*2, k=k_group*2+1
            # b_map[2] = j=j_group*2+1, k=k_group*2
            # b_map[3] = j=j_group*2+1, k=k_group*2+1
            b_map = [
                (j_group * 2) * 8 + k_group * 2,
                (j_group * 2) * 8 + k_group * 2 + 1,
                (j_group * 2 + 1) * 8 + k_group * 2,
                (j_group * 2 + 1) * 8 + k_group * 2 + 1
            ]
            
            # c_map for this 2x2 sub-block (C is 2x2 for these k columns):
            # C[i, k] where i in {0,1}, k in {k_group*2, k_group*2+1}
            # c_map[0] = i=0, k=k_group*2
            # c_map[1] = i=0, k=k_group*2+1
            # c_map[2] = i=1, k=k_group*2
            # c_map[3] = i=1, k=k_group*2+1
            # Flattened C index: k * n + i = k * 2 + i
            c_map = [
                (k_group * 2) * 2 + 0,
                (k_group * 2 + 1) * 2 + 0,
                (k_group * 2) * 2 + 1,
                (k_group * 2 + 1) * 2 + 1
            ]
            
            base_r = (j_group * 4 + k_group) * 7
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val
    
    elapsed = time.time() - start_time
    if elapsed > 140:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
