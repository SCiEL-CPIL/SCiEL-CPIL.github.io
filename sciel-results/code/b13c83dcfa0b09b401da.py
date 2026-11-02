"""
Rank-44 decomposition for 2x3 * 3x8 matrix multiplication.
Uses Strassen's rank-7 algorithm for 2x2x2 blocks plus rank-1 terms for the third A-column.
Total rank: 4 * 11 = 44.
"""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2 * 2x2 using Strassen's rank-7 algorithm.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    r = 7
    U = np.zeros((4, r))
    V = np.zeros((4, r))
    W = np.zeros((4, r))
    
    # Term 0: M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Term 2: M3 = a00(b10-b11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[2, 2] = 1
    
    # Term 3: M4 = a11(b01-b00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 4: M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 5: M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_2x3_3x2_rank11():
    """
    Rank-11 decomposition for 2x3 * 3x2.
    U: shape (6, 11), V: shape (6, 11), W: shape (4, 11)
    Convention: U[i*3+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n, m, p = 2, 3, 2
    r = 11
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # Get Strassen decomposition for 2x2 * 2x2
    Us, Vs, Ws = _strassen_2x2x2()
    
    # Embed Strassen into the larger problem
    U[0, 0:7] = Us[0, :]  # A[0,0]
    U[1, 0:7] = Us[1, :]  # A[0,1]
    U[3, 0:7] = Us[2, :]  # A[1,0]
    U[4, 0:7] = Us[3, :]  # A[1,1]
    
    V[0, 0:7] = Vs[0, :]  # B[0,0]
    V[1, 0:7] = Vs[1, :]  # B[0,1]
    V[2, 0:7] = Vs[2, :]  # B[1,0]
    V[3, 0:7] = Vs[3, :]  # B[1,1]
    
    W[0, 0:7] = Ws[0, :]  # C[0,0]
    W[1, 0:7] = Ws[1, :]  # C[0,1]
    W[2, 0:7] = Ws[2, :]  # C[1,0]
    W[3, 0:7] = Ws[3, :]  # C[1,1]
    
    # Add 4 rank-1 terms for A column 2 with B rows 0,1
    U[2, 7] = 1; V[4, 7] = 1; W[0, 7] = 1  # a02*b20 -> c00
    U[2, 8] = 1; V[5, 8] = 1; W[1, 8] = 1  # a02*b21 -> c01
    U[5, 9] = 1; V[4, 9] = 1; W[2, 9] = 1  # a12*b20 -> c10
    U[5, 10] = 1; V[5, 10] = 1; W[3, 10] = 1  # a12*b21 -> c11
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 10
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Split B (3x8) into 4 blocks of 3x2
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for block in range(4):
        U_b, V_b, W_b = _construct_2x3_3x2_rank11()
        U_blocks.append(U_b)
        V_blocks.append(V_b)
        W_blocks.append(W_b)
    
    r_block = 11
    total_rank = 4 * r_block  # 44
    
    U = [[0.0]*total_rank for _ in range(6)]
    V = [[0.0]*total_rank for _ in range(24)]
    W = [[0.0]*total_rank for _ in range(16)]
    
    for block in range(4):
        offset = block * r_block
        k_start = block * 2  # column offset in B and C
        
        for c in range(r_block):
            for i in range(6):
                U[i][offset + c] = U_blocks[block][i][c]
            for j in range(3):
                for k in range(2):
                    V[j*8 + (k_start + k)][offset + c] = V_blocks[block][j*2 + k][c]
            for k in range(2):
                for i in range(2):
                    W[(k_start + k)*2 + i][offset + c] = W_blocks[block][k*2 + i][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
