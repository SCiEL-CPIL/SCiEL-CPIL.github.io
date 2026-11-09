import numpy as np
import time
import random

DIMENSIONS = [5, 5, 5]

def _strassen_2x2x2():
    """Returns U, V, W for 2x2x2 Strassen decomposition (Rank 7)."""
    # Indices: i in 0..1, j in 0..1, k in 0..1
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen formulas
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U factors (A)
    # Row indices: 00, 01, 10, 11
    U[0, 0] = 1; U[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    U[0, 2] = 1
    U[3, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    U[1, 6] = 1; U[3, 6] = -1
    
    # V factors (B)
    # Row indices: 00, 01, 10, 11
    V[0, 0] = 1; V[3, 0] = 1
    V[0, 1] = 1
    V[1, 2] = 1; V[3, 2] = -1
    V[2, 3] = 1; V[0, 3] = -1
    V[3, 4] = 1
    V[0, 5] = 1; V[1, 5] = 1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W factors (C)
    # Row indices: 00, 01, 10, 11
    W[0, 0] = 1; W[1, 0] = 0; W[2, 0] = 0; W[3, 0] = 1
    W[0, 1] = 0; W[1, 1] = 0; W[2, 1] = 1; W[3, 1] = 1
    W[0, 2] = 0; W[1, 2] = 1; W[2, 2] = 0; W[3, 2] = 0
    W[0, 3] = 1; W[1, 3] = 0; W[2, 3] = 0; W[3, 3] = 1
    W[0, 4] = 0; W[1, 4] = 1; W[2, 4] = 0; W[3, 4] = 1
    W[0, 5] = 1; W[1, 5] = 1; W[2, 5] = 1; W[3, 5] = 0
    W[0, 6] = 0; W[1, 6] = 1; W[2, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """Returns U, V, W for schoolbook decomposition of n x m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _construct_4x4x4_strassen():
    """Constructs 4x4x4 decomposition using 8 Strassen blocks (Rank 56)."""
    U = np.zeros((16, 56))
    V = np.zeros((16, 56))
    W = np.zeros((16, 56))
    
    sU, sV, sW = _strassen_2x2x2()
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # Block at (i, j, k) of size 2x2x2
                u_start = i * 8 + j * 2
                v_start = j * 8 + k * 2
                w_start = k * 8 + i * 2
                
                U[u_start:u_start+4, col:col+7] = sU
                V[v_start:v_start+4, col:col+7] = sV
                W[w_start:w_start+4, col:col+7] = sW
                col += 7
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Partition 5 into 4 and 1
    # Blocks:
    # 4x4x4 (Rank 56)
    # 4x4x1 (Rank 16)
    # 4x1x4 (Rank 16)
    # 1x4x4 (Rank 16)
    # 4x1x1 (Rank 4)
    # 1x4x1 (Rank 4)
    # 1x1x4 (Rank 4)
    # 1x1x1 (Rank 1)
    
    total_rank = 56 + 16 + 16 + 16 + 4 + 4 + 4 + 1
    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    
    col = 0
    
    # 4x4x4
    U4, V4, W4 = _construct_4x4x4_strassen()
    U[0:16, col:col+56] = U4
    V[0:16, col:col+56] = V4
    W[0:16, col:col+56] = W4
    col += 56
    
    # 4x4x1
    U441, V441, W441 = _schoolbook_block(4, 4, 1)
    U[0:16, col:col+16] = U441
    V[0:16, col:col+16] = V441
    W[0:4, col:col+16] = W441 # W is 1x4
    col += 16
    
    # 4x1x4
    U414, V414, W414 = _schoolbook_block(4, 1, 4)
    U[0:4, col:col+16] = U414
    V[0:4, col:col+16] = V414
    W[0:16, col:col+16] = W414 # W is 4x4
    col += 16
    
    # 1x4x4
    U144, V144, W144 = _schoolbook_block(1, 4, 4)
    U[0:4, col:col+16] = U144
    V[0:16, col:col+16] = V144
    W[0:4, col:col+16] = W144 # W is 4x1
    col += 16
    
    # 4x1x1
    U411, V411, W411 = _schoolbook_block(4, 1, 1)
    U[0:4, col:col+4] = U411
    V[0:1, col:col+4] = V411
    W[0:4, col:col+4] = W411
    col += 4
    
    # 1x4x1
    U141, V141, W141 = _schoolbook_block(1, 4, 1)
    U[0:4, col:col+4] = U141
    V[0:4, col:col+4] = V141
    W[0:1, col:col+4] = W141
    col += 4
    
    # 1x1x4
    U114, V114, W114 = _schoolbook_block(1, 1, 4)
    U[0:1, col:col+4] = U114
    V[0:4, col:col+4] = V114
    W[0:4, col:col+4] = W114
    col += 4
    
    # 1x1x1
    U111, V111, W111 = _schoolbook_block(1, 1, 1)
    U[0:1, col:col+1] = U111
    V[0:1, col:col+1] = V111
    W[0:1, col:col+1] = W111
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
