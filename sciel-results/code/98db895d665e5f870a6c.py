import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Hybrid decomposition: schoolbook for i=0 (32 terms) + Strassen for i=1,2 (56 terms)
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]  # (12, 88)
    V = [[0] * rank for _ in range(m * p)]  # (32, 88)
    W = [[0] * rank for _ in range(p * n)]  # (24, 88)
    
    col = 0
    
    # Part 1: i=0, schoolbook for 1x4x8 block
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: i=1,2 using Strassen 2x2x2 blocks
    # Split j into 2 blocks of 2: [0,1] and [2,3]
    # Split k into 4 blocks of 2: [0,1], [2,3], [4,5], [6,7]
    
    # Strassen 2x2x2 local factors (7 multiplications)
    # Local A (2x2): indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local B (2x2): indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local C (2x2): indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_U = [
        [1, 0, 0, 1],   # M1 = (A00+A11)(B00+B11)
        [0, 0, 1, 1],   # M2 = (A10+A11)B00
        [1, 0, 0, 0],   # M3 = A00(B01-B11)
        [0, 0, 0, 1],   # M4 = A11(B10-A00)
        [1, 1, 0, 0],   # M5 = (A00+A01)B11
        [-1, 0, 1, 0],  # M6 = (A10-A00)(B00+B01)
        [0, 1, 0, -1],  # M7 = (A01-A11)(B10+B11)
    ]
    strassen_V = [
        [1, 0, 0, 1],   # B00+B11
        [1, 0, 0, 0],   # B00
        [0, 1, 0, -1],  # B01-B11
        [-1, 0, 1, 0],  # B10-A00
        [0, 0, 0, 1],   # B11
        [1, 1, 0, 0],   # B00+B01
        [0, 0, 1, 1],   # B10+B11
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00+=1, C11+=1
        [0, 0, 1, -1],  # M2: C10+=1, C11-=1
        [0, 1, 0, 1],   # M3: C01+=1, C11+=1
        [1, 0, 1, 0],   # M4: C00+=1, C10+=1
        [-1, 1, 0, 0],  # M5: C00-=1, C01+=1
        [0, 0, 0, 1],   # M6: C11+=1
        [1, 0, 0, 0],   # M7: C00+=1
    ]
    
    # For each j_block (2 blocks) and k_block (4 blocks), apply Strassen
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            
            for t in range(7):
                # U contributions: local A index l -> global (i, j)
                # l=0 -> i'=0, j'=0 -> global i=1, j=j_start+0
                # l=1 -> i'=0, j'=1 -> global i=1, j=j_start+1
                # l=2 -> i'=1, j'=0 -> global i=2, j=j_start+0
                # l=3 -> i'=1, j'=1 -> global i=2, j=j_start+1
                for l in range(4):
                    if strassen_U[t][l] != 0:
                        local_i = l // 2
                        local_j = l % 2
                        global_i = 1 + local_i
                        global_j = j_start + local_j
                        U[global_i * m + global_j][col] += strassen_U[t][l]
                
                # V contributions: local B index l -> global (j, k)
                # l=0 -> j'=0, k'=0 -> global j=j_start+0, k=k_start+0
                # l=1 -> j'=0, k'=1 -> global j=j_start+0, k=k_start+1
                # l=2 -> j'=1, k'=0 -> global j=j_start+1, k=k_start+0
                # l=3 -> j'=1, k'=1 -> global j=j_start+1, k=k_start+1
                for l in range(4):
                    if strassen_V[t][l] != 0:
                        local_j = l // 2
                        local_k = l % 2
                        global_j = j_start + local_j
                        global_k = k_start + local_k
                        V[global_j * p + global_k][col] += strassen_V[t][l]
                
                # W contributions: local C index l -> global (k, i)
                # l=0 -> k'=0, i'=0 -> global k=k_start+0, i=1
                # l=1 -> k'=0, i'=1 -> global k=k_start+0, i=2
                # l=2 -> k'=1, i'=0 -> global k=k_start+1, i=1
                # l=3 -> k'=1, i'=1 -> global k=k_start+1, i=2
                for l in range(4):
                    if strassen_W[t][l] != 0:
                        local_k = l // 2
                        local_i = l % 2
                        global_k = k_start + local_k
                        global_i = 1 + local_i
                        W[global_k * n + global_i][col] += strassen_W[t][l]
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
