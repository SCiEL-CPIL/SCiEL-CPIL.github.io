"""Rewritten program for matrix multiplication 3,4,8 using rank-88 decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 88
    
    # Initialize with None or 0? The interface expects matrices.
    # Using lists for exact integer/half-integer as requested, but np is available.
    # The prompt says "integer/half-integer entries". 
    # Standard python lists are safe for exactness.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i=0 (1x4x8 block)
    # Schoolbook: 32 terms
    # C_k = sum_j A_j B_jk
    # Tensr: U_j (1x4), V_jk (4x8), W_k (8x1? No, W is p*n = 8x3)
    # W[k*n + i]
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1  # i=0, j=j
            V[j * p + k][col] = 1  # j=j, k=k
            W[k * n + 0][col] = 1  # k=k, i=0
            col += 1
            
    # Part 2: i=1,2 (2x4x8 block)
    # Split j into 2 blocks of 2: J0=[0,1], J1=[2,3]
    # Split k into 4 blocks of 2: K0=[0,1], K1=[2,3], K2=[4,5], K3=[6,7]
    
    # Strassen 2x2x2 factors
    # Local indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B10-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B00-B10) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B01+B00) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    # Note: The standard Strassen formulas vary. Using a specific valid set.
    # Let's use the set from the previous code which was correct.
    
    sU = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2 (A10+A11) -> local 2,3
        [1, 0, 0, 0], # M3 (A00)
        [0, 0, 0, 1], # M4 (A11)
        [1, 1, 0, 0], # M5 (A00+A01)
        [-1, 0, 1, 0], # M6 (A10-A00)
        [0, 1, 0, -1]  # M7 (A01-A11)
    ]
    sV = [
        [1, 0, 0, 1], # M1 (B00+B11)
        [1, 0, 0, 0], # M2 (B00)
        [0, 1, 0, -1], # M3 (B10-B11) -> local 2,3
        [-1, 0, 1, 0], # M4 (B00-B10) -> local 0,2
        [0, 0, 0, 1], # M5 (B11)
        [1, 1, 0, 0], # M6 (B01+B00) -> local 0,1
        [0, 0, 1, 1]  # M7 (B10+B11) -> local 2,3
    ]
    sW = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 0, 1, -1], # M2 -> C10, C11-
        [0, 1, 0, 1], # M3 -> C01, C11
        [1, 0, 1, 0], # M4 -> C00, C10
        [-1, 1, 0, 0], # M5 -> C00-, C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]

    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            
            for t in range(7):
                # U updates
                # Global i: 1, 2 (indices 1+0, 1+1)
                # Local A index l:
                # l=0: i'=0, j'=0 -> i=1, j=j_start+0
                # l=1: i'=0, j'=1 -> i=1, j=j_start+1
                # l=2: i'=1, j'=0 -> i=2, j=j_start+0
                # l=3: i'=1, j'=1 -> i=2, j=j_start+1
                
                for l in range(4):
                    val = sU[t][l]
                    if val != 0:
                        local_i = l // 2
                        local_j = l % 2
                        g_i = 1 + local_i
                        g_j = j_start + local_j
                        U[g_i * m + g_j][col] += val
                        
                # V updates
                # Global j, k
                # Local B index l:
                # l=0: j'=0, k'=0 -> j=j_start+0, k=k_start+0
                # l=1: j'=0, k'=1 -> j=j_start+0, k=k_start+1
                # l=2: j'=1, k'=0 -> j=j_start+1, k=k_start+0
                # l=3: j'=1, k'=1 -> j=j_start+1, k=k_start+1
                
                for l in range(4):
                    val = sV[t][l]
                    if val != 0:
                        local_j = l // 2
                        local_k = l % 2
                        g_j = j_start + local_j
                        g_k = k_start + local_k
                        V[g_j * p + g_k][col] += val
                        
                # W updates
                # Global k, i
                # Local C index l:
                # l=0: i'=0, k'=0 -> i=1, k=k_start+0
                # l=1: i'=0, k'=1 -> i=1, k=k_start+1
                # l=2: i'=1, k'=0 -> i=2, k=k_start+0
                # l=3: i'=1, k'=1 -> i=2, k=k_start+1
                
                for l in range(4):
                    val = sW[t][l]
                    if val != 0:
                        local_i = l // 2
                        local_k = l % 2
                        g_i = 1 + local_i
                        g_k = k_start + local_k
                        W[g_k * n + g_i][col] += val
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
