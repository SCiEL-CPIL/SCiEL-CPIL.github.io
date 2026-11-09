import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # We will use a Strassen-like reduction on a 2x2x2 sub-block.
    # Block indices: i in {0, 1}, j in {0, 1}, k in {0, 1}
    # This reduces the rank of this sub-block from 8 to 7.
    # The remaining terms are handled by schoolbook.
    
    # Total rank will be 105 - 8 + 7 = 104.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term
    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
    # Initialize zero vectors
    size_a = n * m
    size_b = m * p
    size_c = p * n
    
    # Add schoolbook terms for all (i, j, k) NOT in the 2x2x2 block
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue # Skip the block we will handle with Strassen
                
                u = [0] * size_a
                v = [0] * size_b
                w = [0] * size_c
                
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                
                u[a_idx] = 1
                v[b_idx] = 1
                w[c_idx] = 1
                
                add_term(u, v, w)
                
    # Add Strassen terms for the 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # Indices in the block:
    # A_00 (0,0), A_01 (0,1), A_10 (1,0), A_11 (1,1)
    # B_00 (0,0), B_01 (0,1), B_10 (1,0), B_11 (1,1)
    # C_00 (0,0), C_01 (0,1), C_10 (1,0), C_11 (1,1)
    
    # M1 = A_00 * (B_00 + B_11)
    u1 = [0] * size_a; v1 = [0] * size_b; w1 = [0] * size_c
    u1[0 * m + 0] = 1
    v1[0 * p + 0] = 1
    v1[1 * p + 1] = 1
    w1[0 * n + 0] = 1 # C_00
    w1[1 * n + 1] = 1 # C_11
    add_term(u1, v1, w1)
    
    # M2 = (A_00 + A_01) * B_11
    u2 = [0] * size_a; v2 = [0] * size_b; w2 = [0] * size_c
    u2[0 * m + 0] = 1
    u2[0 * m + 1] = 1
    v2[1 * p + 1] = 1
    w2[0 * n + 1] = 1 # C_01
    w2[1 * n + 1] = -1 # C_11
    add_term(u2, v2, w2)
    
    # M3 = (A_10 + A_11) * B_00
    u3 = [0] * size_a; v3 = [0] * size_b; w3 = [0] * size_c
    u3[1 * m + 0] = 1
    u3[1 * m + 1] = 1
    v3[0 * p + 0] = 1
    w3[1 * n + 0] = 1 # C_10
    w3[1 * n + 1] = 1 # C_11
    add_term(u3, v3, w3)
    
    # M4 = A_11 * (B_01 + B_10)
    u4 = [0] * size_a; v4 = [0] * size_b; w4 = [0] * size_c
    u4[1 * m + 1] = 1
    v4[0 * p + 1] = 1
    v4[1 * p + 0] = 1
    w4[0 * n + 0] = 1 # C_00
    w4[1 * n + 1] = 1 # C_11
    add_term(u4, v4, w4)
    
    # M5 = (A_00 + A_11) * (B_11 + B_10)
    u5 = [0] * size_a; v5 = [0] * size_b; w5 = [0] * size_c
    u5[0 * m + 0] = 1
    u5[1 * m + 1] = 1
    v5[1 * p + 1] = 1
    v5[1 * p + 0] = 1
    w5[0 * n + 0] = -1 # C_00
    w5[0 * n + 1] = 1  # C_01
    w5[1 * n + 1] = -1 # C_11
    add_term(u5, v5, w5)
    
    # M6 = (A_10 + A_00) * (B_01 + B_00)
    u6 = [0] * size_a; v6 = [0] * size_b; w6 = [0] * size_c
    u6[1 * m + 0] = 1
    u6[0 * m + 0] = 1
    v6[0 * p + 1] = 1
    v6[0 * p + 0] = 1
    w6[1 * n + 0] = 1 # C_10
    add_term(u6, v6, w6)
    
    # M7 = (A_01 + A_11) * (B_10 + B_11)
    u7 = [0] * size_a; v7 = [0] * size_b; w7 = [0] * size_c
    u7[0 * m + 1] = 1
    u7[1 * m + 1] = 1
    v7[1 * p + 0] = 1
    v7[1 * p + 1] = 1
    w7[0 * n + 0] = -1 # C_00
    w7[1 * n + 0] = 1  # C_10
    add_term(u7, v7, w7)
    
    rank = len(U_list)
    
    # Convert to the required format
    # U, V, W are lists of lists
    # U has shape (size_a, rank)
    # We currently have U_list as list of vectors (size_a)
    # We need to transpose.
    
    U = [[0] * rank for _ in range(size_a)]
    V = [[0] * rank for _ in range(size_b)]
    W = [[0] * rank for _ in range(size_c)]
    
    for r in range(rank):
        for i in range(size_a):
            U[i][r] = U_list[r][i]
        for i in range(size_b):
            V[i][r] = V_list[r][i]
        for i in range(size_c):
            W[i][r] = W_list[r][i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
