"""Matrix multiplication 2,2,8 - rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_u = n * m  # 4
    num_v = m * p  # 16
    num_w = p * n  # 16
    
    # Use Strassen decomposition for each 2x2 x 2x2 block
    num_blocks = p // 2  # 4
    strassen_rank = 7
    rank = num_blocks * strassen_rank  # 28
    
    U = np.zeros((num_u, rank), dtype=float)
    V = np.zeros((num_v, rank), dtype=float)
    W = np.zeros((num_w, rank), dtype=float)
    
    col = 0
    for block in range(num_blocks):
        k_offset = block * 2
        
        # M1: (a11+a22)(b11+b22)
        # U: a11=1, a22=1
        U[0, col] = 1
        U[3, col] = 1
        # V: b11=1, b22=1
        V[0*p + k_offset, col] = 1
        V[1*p + k_offset+1, col] = 1
        # W: c11=+1, c22=+1
        W[k_offset*n + 0, col] = 1
        W[(k_offset+1)*n + 1, col] = 1
        col += 1
        
        # M2: (a21+a22)b11
        # U: a21=1, a22=1
        U[2, col] = 1
        U[3, col] = 1
        # V: b11=1
        V[0*p + k_offset, col] = 1
        # W: c21=+1, c22=-1
        W[k_offset*n + 1, col] = 1
        W[(k_offset+1)*n + 1, col] = -1
        col += 1
        
        # M3: a11(b21-b22)
        # U: a11=1
        U[0, col] = 1
        # V: b21=1, b22=-1
        V[1*p + k_offset, col] = 1
        V[1*p + k_offset+1, col] = -1
        # W: c12=+1, c22=+1
        W[(k_offset+1)*n + 0, col] = 1
        W[(k_offset+1)*n + 1, col] = 1
        col += 1
        
        # M4: a22(b12-b11)
        # U: a22=1
        U[3, col] = 1
        # V: b12=1, b11=-1
        V[0*p + k_offset+1, col] = 1
        V[0*p + k_offset, col] = -1
        # W: c11=+1, c21=+1
        W[k_offset*n + 0, col] = 1
        W[k_offset*n + 1, col] = 1
        col += 1
        
        # M5: (a11+a12)b22
        # U: a11=1, a12=1
        U[0, col] = 1
        U[1, col] = 1
        # V: b22=1
        V[1*p + k_offset+1, col] = 1
        # W: c11=-1, c12=+1
        W[k_offset*n + 0, col] = -1
        W[(k_offset+1)*n + 0, col] = 1
        col += 1
        
        # M6: (a21-a11)(b11+b21)
        # U: a21=1, a11=-1
        U[2, col] = 1
        U[0, col] = -1
        # V: b11=1, b21=1
        V[0*p + k_offset, col] = 1
        V[1*p + k_offset, col] = 1
        # W: c22=+1
        W[(k_offset+1)*n + 1, col] = 1
        col += 1
        
        # M7: (a12-a22)(b22+b12)
        # U: a12=1, a22=-1
        U[1, col] = 1
        U[3, col] = -1
        # V: b22=1, b12=1
        V[1*p + k_offset+1, col] = 1
        V[0*p + k_offset+1, col] = 1
        # W: c11=+1
        W[k_offset*n + 0, col] = 1
        col += 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
