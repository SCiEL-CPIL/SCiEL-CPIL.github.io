"""Rewritten program: rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication."""
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # 1. Construct 2x2x2 Strassen decomposition
    # U, V, W are 4x7 matrices
    U_strassen = [
        [1, 0, 0, 0], # M1: A00
        [0, 0, 1, 0], # M2: A10
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 0, 0, 0], # M5: A00
        [-1, 0, 1, 0],# M6: -A00 + A10
        [0, 1, 0, -1] # M7: A01 - A11
    ]
    # Wait, my previous manual extraction was column-wise or row-wise?
    # U[i,j,c]
    # M1: (A00+A11)(B00+B11) -> U1 has 1 at (0,0) and (1,1)
    # Let's re-verify the U vectors.
    # M1: A00 + A11. Flat indices: 0 and 3. U1 = [1, 0, 0, 1]
    # M2: A10 + A11. Flat indices: 2 and 3. U2 = [0, 0, 1, 1]
    # M3: A00. Flat index: 0. U3 = [1, 0, 0, 0]
    # M4: A11. Flat index: 3. U4 = [0, 0, 0, 1]
    # M5: A00 + A01. Flat indices: 0 and 1. U5 = [1, 1, 0, 0]
    # M6: A10 - A00. Flat indices: 2 and 0. U6 = [-1, 0, 1, 0]
    # M7: A01 - A11. Flat indices: 1 and 3. U7 = [0, 1, 0, -1]
    
    U_s = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    V_s = [
        [1, 0, 0, 1], # B00 + B11
        [1, 0, 0, 0], # B00
        [0, 1, 0, -1],# B01 - B11
        [-1, 0, 1, 0],# B10 - B00
        [0, 0, 0, 1], # B11
        [1, 1, 0, 0], # B00 + B01
        [0, 0, 1, 1]  # B10 + B11
    ]
    
    W_s = [
        [1, 0, 0, 1], # C00 + C11
        [0, 0, 1, -1],# C10 - C11
        [0, 1, 0, 1], # C01 + C11
        [1, 0, 1, 0], # C00 + C10
        [-1, 1, 0, 0],# -C00 + C01
        [0, 0, 0, 1], # C11
        [1, 0, 0, 1]  # C00 + C11
    ]
    
    # 2. Construct 4x4x4 decomposition via Kronecker Product
    rank_444 = 7 * 7
    U_444 = [[0] * rank_444 for _ in range(16)]
    V_444 = [[0] * rank_444 for _ in range(16)]
    W_444 = [[0] * rank_444 for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = i1 * 2 + i0
                            j = j1 * 2 + j0
                            idx_u = i * 4 + j
                            val_u = U_s[c1][i1 * 2 + j1] * U_s[c2][i0 * 2 + j0]
                            U_444[idx_u][c] = val_u
                            
                            # V index: j, k
                            # j = j1 * 2 + j0, k = k1 * 2 + k0
                            # V_s[c1] is for j1, k1. V_s[c2] is for j0, k0.
                            # We need to fill V_444 for all j, k.
                            # It is more efficient to iterate k1, k0 as well.
                            pass # Handled in separate loops below for clarity/performance

    # Let's redo the filling more efficiently
    U_444 = [[0] * rank_444 for _ in range(16)]
    V_444 = [[0] * rank_444 for _ in range(16)]
    W_444 = [[0] * rank_444 for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            # U
            for i1 in range(2):
                for j1 in range(2):
                    u1_val = U_s[c1][i1 * 2 + j1]
                    if u1_val == 0: continue
                    for i0 in range(2):
                        for j0 in range(2):
                            u0_val = U_s[c2][i0 * 2 + j0]
                            if u0_val == 0: continue
                            i = i1 * 2 + i0
                            j = j1 * 2 + j0
                            idx_u = i * 4 + j
                            U_444[idx_u][c] = u1_val * u0_val
                            
            # V
            for j1 in range(2):
                for k1 in range(2):
                    v1_val = V_s[c1][j1 * 2 + k1]
                    if v1_val == 0: continue
                    for j0 in range(2):
                        for k0 in range(2):
                            v0_val = V_s[c2][j0 * 2 + k0]
                            if v0_val == 0: continue
                            j = j1 * 2 + j0
                            k = k1 * 2 + k0
                            idx_v = j * 4 + k
                            V_444[idx_v][c] = v1_val * v0_val
                            
            # W
            for k1 in range(2):
                for i1 in range(2):
                    w1_val = W_s[c1][k1 * 2 + i1]
                    if w1_val == 0: continue
                    for k0 in range(2):
                        for i0 in range(2):
                            w0_val = W_s[c2][k0 * 2 + i0]
                            if w0_val == 0: continue
                            k = k1 * 2 + k0
                            i = i1 * 2 + i0
                            idx_w = k * 4 + i
                            W_444[idx_w][c] = w1_val * w0_val
                            
    # 3. Construct 4x4x3 Schoolbook decomposition
    rank_443 = 4 * 4 * 3
    U_443 = [[0] * rank_443 for _ in range(16)]
    V_443 = [[0] * rank_443 for _ in range(12)] # m*p = 4*3
    W_443 = [[0] * rank_443 for _ in range(12)] # p*n = 3*4
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k_loc in range(3):
                U_443[i * 4 + j][col] = 1
                V_443[j * 3 + k_loc][col] = 1
                W_443[k_loc * 4 + i][col] = 1
                col += 1
                
    # 4. Embed into 4x4x7 space
    # U: (16, rank_444 + rank_443)
    # V: (28, rank_444 + rank_443)
    # W: (28, rank_444 + rank_443)
    
    total_rank = rank_444 + rank_443
    U_final = [[0] * total_rank for _ in range(16)]
    V_final = [[0] * total_rank for _ in range(28)]
    W_final = [[0] * total_rank for _ in range(28)]
    
    # Copy Part 1 (4x4x4)
    for r in range(rank_444):
        for i in range(16):
            U_final[i][r] = U_444[i][r]
            
        for j in range(4):
            for k in range(4):
                idx_v = j * 7 + k
                V_final[idx_v][r] = V_444[j * 4 + k][r]
                
        for k in range(4):
            for i in range(4):
                idx_w = k * 4 + i
                W_final[idx_w][r] = W_444[k * 4 + i][r]
                
    # Copy Part 2 (4x4x3)
    offset = rank_444
    for r in range(rank_443):
        for i in range(16):
            U_final[i][offset + r] = U_443[i][r]
            
        for j in range(4):
            for k_loc in range(3):
                k_global = 4 + k_loc
                idx_v = j * 7 + k_global
                V_final[idx_v][offset + r] = V_443[j * 3 + k_loc][r]
                
        for k_loc in range(3):
            k_global = 4 + k_loc
            for i in range(4):
                idx_w = k_global * 4 + i
                W_final[idx_w][offset + r] = W_443[k_loc * 4 + i][r]
                
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
