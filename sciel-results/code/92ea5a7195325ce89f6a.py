import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(rank_offset, j_offset, k_offset):
        a_idx = 0 * m + j_offset
        b_idx = 0 * m + (j_offset + 1)
        c_idx = 1 * m + j_offset
        d_idx = 1 * m + (j_offset + 1)
        
        e_idx = j_offset * p + k_offset
        f_idx = j_offset * p + (k_offset + 1)
        g_idx = (j_offset + 1) * p + k_offset
        h_idx = (j_offset + 1) * p + (k_offset + 1)
        
        r0 = rank_offset
        r1 = rank_offset + 1
        r2 = rank_offset + 2
        r3 = rank_offset + 3
        r4 = rank_offset + 4
        r5 = rank_offset + 5
        r6 = rank_offset + 6
        
        U[a_idx][r0] += 1; U[d_idx][r0] += 1
        U[c_idx][r1] += 1; U[d_idx][r1] += 1
        U[a_idx][r2] += 1
        U[d_idx][r3] += 1
        U[a_idx][r4] += 1; U[b_idx][r4] += 1
        U[c_idx][r5] += 1; U[a_idx][r5] -= 1
        U[b_idx][r6] += 1; U[d_idx][r6] -= 1
        
        V[e_idx][r0] += 1; V[h_idx][r0] += 1
        V[e_idx][r1] += 1
        V[f_idx][r2] += 1; V[h_idx][r2] -= 1
        V[g_idx][r3] += 1; V[e_idx][r3] -= 1
        V[h_idx][r4] += 1
        V[e_idx][r5] += 1; V[f_idx][r5] += 1
        V[g_idx][r6] += 1; V[h_idx][r6] += 1
        
        W[k_offset * n + 0][r0] += 1
        W[k_offset * n + 0][r3] += 1
        W[k_offset * n + 0][r4] -= 1
        W[k_offset * n + 0][r6] += 1
        
        W[(k_offset + 1) * n + 0][r2] += 1
        W[(k_offset + 1) * n + 0][r4] += 1
        
        W[k_offset * n + 1][r1] += 1
        W[k_offset * n + 1][r3] += 1
        
        W[(k_offset + 1) * n + 1][r0] += 1
        W[(k_offset + 1) * n + 1][r2] += 1
        W[(k_offset + 1) * n + 1][r5] += 1
        W[(k_offset + 1) * n + 1][r6] += 1
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    add_strassen_block(0, 0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    add_strassen_block(7, 0, 2)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] -> C[:, 4:5] (schoolbook 2x2x1)
    U[0][14] += 1; V[0*5+4][14] += 1; W[4*2+0][14] += 1
    U[1][15] += 1; V[1*5+4][15] += 1; W[4*2+0][15] += 1
    U[3][16] += 1; V[0*5+4][16] += 1; W[4*2+1][16] += 1
    U[4][17] += 1; V[1*5+4][17] += 1; W[4*2+1][17] += 1
    
    # Block 4: A[0:2, 2:3] * B[2:3, 0:5] -> C[:, 0:5] (schoolbook 2x1x5)
    for k in range(5):
        U[2][18 + k] += 1
        V[2*5 + k][18 + k] += 1
        W[k*2 + 0][18 + k] += 1
        U[5][23 + k] += 1
        V[2*5 + k][23 + k] += 1
        W[k*2 + 1][23 + k] += 1
    
    return {"U": U, "V": V, "W": W}
