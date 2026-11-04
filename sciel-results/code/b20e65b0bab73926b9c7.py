"""Program for rank-minimal exact tensor decomposition of 4x5*5x6 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2 * 2x2 -> 2x2 multiplication with rank 7.
    Indices:
    A: 2x2 -> 4 elements. Index a = i*2 + j.
    B: 2x2 -> 4 elements. Index b = j*2 + k.
    C: 2x2 -> 4 elements. Index c = k*2 + i (Trace convention W[k*n+i, r] with n=2).
    Note: The task uses W[k*n+i, r]. For 2x2, n=2.
    C[i, k] corresponds to W[k*2 + i, r].
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Strassen's 7 multiplications:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # A indices: 00:0, 01:1, 10:2, 11:3
    # B indices: 00:0, 01:1, 10:2, 11:3
    # C indices (W): C00 (k=0,i=0):0, C01 (k=1,i=0):1, C10 (k=0,i=1):2, C11 (k=1,i=1):3
    # W[k*2 + i, r]
    # C00: W[0, r]
    # C01: W[1, r]
    # C10: W[2, r]
    # C11: W[3, r]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = -1; W[2][0] = -1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[0][1] = 0; W[1][1] = 0; W[2][1] = 1; W[3][1] = 1
    
    # M3
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0][2] = 0; W[1][2] = 0; W[2][2] = 0; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 0; W[2][3] = 0; W[3][3] = 0
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 0; W[1][4] = 1; W[2][4] = 0; W[3][4] = 0
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 0; W[1][5] = 0; W[2][5] = 1; W[3][5] = -1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = -1; W[2][6] = 0; W[3][6] = 0
    
    return U, V, W

def _decompose_2x2x6():
    """
    Decomposes 2x2 * 2x6 -> 2x6.
    Splits B into three 2x2 blocks.
    Rank = 3 * 7 = 21.
    A: 2x2 (4 elems). Index a = i*2 + j.
    B: 2x6 (12 elems). Index b = j*6 + k.
    C: 2x6 (12 elems). Index c = k*2 + i.
    """
    rank = 21
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(12)]
    W = [[0]*rank for _ in range(12)]
    
    # B is 2x6. Split into B0 (2x2), B1 (2x2), B2 (2x2).
    # B0: cols 0,1. B1: cols 2,3. B2: cols 4,5.
    # B index: j*6 + k.
    # B0: j=0,1; k=0,1. Indices: 0,1,6,7.
    # B1: j=0,1; k=2,3. Indices: 2,3,8,9.
    # B2: j=0,1; k=4,5. Indices: 4,5,10,11.
    
    # For each block, use Strassen 2x2x2.
    # The A part is the same for all blocks.
    # We need to map the local Strassen U,V,W to the global indices.
    
    U_str, V_str, W_str = _strassen_2x2x2()
    
    for block_idx in range(3):
        base_r = block_idx * 7
        # Map V indices: local j, k_local -> global j, k_global
        # k_global = block_idx * 2 + k_local
        for r in range(7):
            # U is same
            for a in range(4):
                U[a][base_r + r] = U_str[a][r]
            
            # V mapping
            # local b = j*2 + k_local
            # global b = j*6 + k_global = j*6 + block_idx*2 + k_local
            for j in range(2):
                for k_local in range(2):
                    b_local = j*2 + k_local
                    b_global = j*6 + block_idx*2 + k_local
                    V[b_global][base_r + r] = V_str[b_local][r]
            
            # W mapping
            # local c = k_local*2 + i
            # global c = k_global*2 + i = (block_idx*2 + k_local)*2 + i
            for i in range(2):
                for k_local in range(2):
                    c_local = k_local*2 + i
                    c_global = (block_idx*2 + k_local)*2 + i
                    W[c_global][base_r + r] = W_str[c_local][r]
    
    return U, V, W

def _decompose_2x1x6():
    """
    Decomposes 2x1 * 1x6 -> 2x6.
    Rank = 12 (Schoolbook, as no Strassen benefit for 1x1 middle).
    A: 2x1 (2 elems). Index a = i*1 + 0 = i.
    B: 1x6 (6 elems). Index b = 0*6 + k = k.
    C: 2x6 (12 elems). Index c = k*2 + i.
    """
    rank = 12
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(12)]
    
    col = 0
    for i in range(2):
        for k in range(6):
            U[i][col] = 1
            V[k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
            
    return U, V, W

def _decompose_2x3x6():
    """
    Decomposes 2x3 * 3x6 -> 2x6.
    Splits middle dim 3 into 1+2.
    A: 2x3. Split into A1 (2x1) and A2 (2x2).
    B: 3x6. Split into B1 (1x6) and B2 (2x6).
    C = A1 B1 + A2 B2.
    Rank = Rank(2x1x6) + Rank(2x2x6) = 12 + 21 = 33.
    A indices: 0..5 (i*3+j).
    B indices: 0..17 (j*6+k).
    C indices: 0..11 (k*2+i).
    """
    rank = 33
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(18)]
    W = [[0]*rank for _ in range(12)]
    
    # Part 1: A1 (2x1) * B1 (1x6)
    U1, V1, W1 = _decompose_2x1x6()
    # Map A1 (2x1) to A (2x3) cols 0
    # A1 index a1 = i. A index a = i*3 + 0.
    for r in range(12):
        for i in range(2):
            U[i*3 + 0][r] = U1[i][r]
        # Map B1 (1x6) to B (3x6) rows 0
        # B1 index b1 = k. B index b = 0*6 + k = k.
        for k in range(6):
            V[k][r] = V1[k][r]
        # W is same (2x6)
        for c in range(12):
            W[c][r] = W1[c][r]
            
    # Part 2: A2 (2x2) * B2 (2x6)
    U2, V2, W2 = _decompose_2x2x6()
    offset = 12
    # Map A2 (2x2) to A (2x3) cols 1,2
    # A2 index a2 = i*2 + j_local (j_local 0,1).
    # A index a = i*3 + (1 + j_local).
    for r in range(21):
        for i in range(2):
            for j_local in range(2):
                a2 = i*2 + j_local
                a = i*3 + 1 + j_local
                U[a][offset + r] = U2[a2][r]
        
        # Map B2 (2x6) to B (3x6) rows 1,2
        # B2 index b2 = j_local*6 + k (j_local 0,1).
        # B index b = (1 + j_local)*6 + k.
        for j_local in range(2):
            for k in range(6):
                b2 = j_local*6 + k
                b = (1 + j_local)*6 + k
                V[b][offset + r] = V2[b2][r]
        
        # W is same (2x6)
        for c in range(12):
            W[c][offset + r] = W2[c][r]
            
    return U, V, W

def _decompose_4x2x6():
    """
    Decomposes 4x2 * 2x6 -> 4x6.
    Splits A (4x2) into two 2x2 blocks vertically.
    A1 (2x2), A2 (2x2).
    B is 2x6.
    C1 = A1 B (2x6), C2 = A2 B (2x6).
    Rank = 2 * Rank(2x2x6) = 2 * 21 = 42.
    A indices: 0..7 (i*2+j).
    B indices: 0..11 (j*6+k).
    C indices: 0..23 (k*4+i).
    """
    rank = 42
    U = [[0]*rank for _ in range(8)]
    V = [[0]*rank for _ in range(12)]
    W = [[0]*rank for _ in range(24)]
    
    U1, V1, W1 = _decompose_2x2x6()
    
    for block_idx in range(2):
        base_r = block_idx * 21
        # A1/A2 are 2x2.
        # A index: block_idx*4 + i*2 + j.
        for r in range(21):
            for i in range(2):
                for j in range(2):
                    a_local = i*2 + j
                    a_global = block_idx*4 + a_local
                    U[a_global][base_r + r] = U1[a_local][r]
            
            # V is same B (2x6)
            for b in range(12):
                V[b][base_r + r] = V1[b][r]
            
            # W mapping: C is 4x6.
            # C1 is top 2 rows, C2 is bottom 2 rows.
            # W index: k*4 + i.
            # For block 0, i is 0,1. For block 1, i is 2,3.
            for i in range(2):
                for k in range(6):
                    c_local = k*2 + i
                    c_global = k*4 + (block_idx*2 + i)
                    W[c_global][base_r + r] = W1[c_local][r]
    
    return U, V, W

def _decompose_4x3x6():
    """
    Decomposes 4x3 * 3x6 -> 4x6.
    Splits A (4x3) into two 2x3 blocks vertically.
    A1 (2x3), A2 (2x3).
    B is 3x6.
    C1 = A1 B (2x6), C2 = A2 B (2x6).
    Rank = 2 * Rank(2x3x6) = 2 * 33 = 66.
    A indices: 0..11 (i*3+j).
    B indices: 0..17 (j*6+k).
    C indices: 0..23 (k*4+i).
    """
    rank = 66
    U = [[0]*rank for _ in range(12)]
    V = [[0]*rank for _ in range(18)]
    W = [[0]*rank for _ in range(24)]
    
    U1, V1, W1 = _decompose_2x3x6()
    
    for block_idx in range(2):
        base_r = block_idx * 33
        for r in range(33):
            for i in range(2):
                for j in range(3):
                    a_local = i*3 + j
                    a_global = block_idx*6 + a_local
                    U[a_global][base_r + r] = U1[a_local][r]
            
            for b in range(18):
                V[b][base_r + r] = V1[b][r]
            
            for i in range(2):
                for k in range(6):
                    c_local = k*2 + i
                    c_global = k*4 + (block_idx*2 + i)
                    W[c_global][base_r + r] = W1[c_local][r]
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    Uses block decomposition:
    Split middle dim 5 into 2+3.
    A = [A_L (4x2), A_R (4x3)]
    B = [B_T (2x6); B_B (3x6)]
    C = A_L B_T + A_R B_B
    
    Rank(A_L B_T) = Rank(4x2x6) = 42
    Rank(A_R B_B) = Rank(4x3x6) = 66
    Total Rank = 108
    """
    n, m, p = 4, 5, 6
    # A: 4x5, B: 5x6, C: 4x6
    # A indices: i*5+j
    # B indices: j*6+k
    # C indices: k*4+i
    
    U_L, V_L, W_L = _decompose_4x2x6() # 4x2 * 2x6 -> 4x6. Rank 42.
    U_R, V_R, W_R = _decompose_4x3x6() # 4x3 * 3x6 -> 4x6. Rank 66.
    
    rank = 108
    U = [[0]*rank for _ in range(20)]
    V = [[0]*rank for _ in range(30)]
    W = [[0]*rank for _ in range(24)]
    
    # Map U_L (4x2) to A (4x5) cols 0,1
    # U_L index: i*2+j. A index: i*5+j.
    for r in range(42):
        for i in range(4):
            for j in range(2):
                u_l_idx = i*2 + j
                a_idx = i*5 + j
                U[a_idx][r] = U_L[u_l_idx][r]
        
        # Map V_L (2x6) to B (5x6) rows 0,1
        # V_L index: j*6+k. B index: j*6+k. (Same for j=0,1)
        for j in range(2):
            for k in range(6):
                v_l_idx = j*6 + k
                b_idx = j*6 + k
                V[b_idx][r] = V_L[v_l_idx][r]
        
        # Map W_L (4x6) to C (4x6)
        # W_L index: k*4+i. C index: k*4+i.
        for c in range(24):
            W[c][r] = W_L[c][r]
            
    # Map U_R (4x3) to A (4x5) cols 2,3,4
    # U_R index: i*3+j. A index: i*5 + (2+j).
    offset = 42
    for r in range(66):
        for i in range(4):
            for j in range(3):
                u_r_idx = i*3 + j
                a_idx = i*5 + 2 + j
                U[a_idx][offset + r] = U_R[u_r_idx][r]
        
        # Map V_R (3x6) to B (5x6) rows 2,3,4
        # V_R index: j*6+k. B index: (2+j)*6+k.
        for j in range(3):
            for k in range(6):
                v_r_idx = j*6 + k
                b_idx = (2+j)*6 + k
                V[b_idx][offset + r] = V_R[v_r_idx][r]
        
        # Map W_R (4x6) to C (4x6)
        for c in range(24):
            W[c][offset + r] = W_R[c][r]
            
    return {"U": U, "V": V, "W": W}
