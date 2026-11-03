import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # A is 2x4, B is 4x8, C is 2x8
    
    # Rank 56
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    col = 0
    for s in range(2): # s=0 (A_L, B_top), s=1 (A_R, B_bot)
        for j in range(4): # j=0..3 (col blocks of B)
            if time.time() > deadline:
                break
            
            # Strassen for A_s (2x2) and B_{s,j} (2x2)
            
            # A_s indices in global A (2x4)
            # A_s is at cols 2s, 2s+1
            # Rows 0, 1
            a_idx = {
                (0,0): 0 * m + (2*s + 0),
                (0,1): 0 * m + (2*s + 1),
                (1,0): 1 * m + (2*s + 0),
                (1,1): 1 * m + (2*s + 1)
            }
            
            # B_{s,j} indices in global B (4x8)
            # B_{s,j} is at rows 2s, 2s+1 and cols 2j, 2j+1
            b_idx = {
                (0,0): (2*s + 0) * p + (2*j + 0),
                (0,1): (2*s + 0) * p + (2*j + 1),
                (1,0): (2*s + 1) * p + (2*j + 0),
                (1,1): (2*s + 1) * p + (2*j + 1)
            }
            
            # C contribution indices in global C (2x8)
            # Local C (2x2) maps to global C rows 0,1 and cols 2j, 2j+1
            # Global C index: k*n + i
            c_idx = {
                (0,0): (2*j + 0) * n + 0,
                (0,1): (2*j + 1) * n + 0,
                (1,0): (2*j + 0) * n + 1,
                (1,1): (2*j + 1) * n + 1
            }
            
            # Strassen U coefficients
            # u1 = a00 + a11
            U[a_idx[(0,0)]][col] += 1
            U[a_idx[(1,1)]][col] += 1
            # u2 = a10 + a11
            U[a_idx[(1,0)]][col+1] += 1
            U[a_idx[(1,1)]][col+1] += 1
            # u3 = a00
            U[a_idx[(0,0)]][col+2] += 1
            # u4 = a11
            U[a_idx[(1,1)]][col+3] += 1
            # u5 = a00 + a01
            U[a_idx[(0,0)]][col+4] += 1
            U[a_idx[(0,1)]][col+4] += 1
            # u6 = a10 - a00
            U[a_idx[(1,0)]][col+5] += 1
            U[a_idx[(0,0)]][col+5] += -1
            # u7 = a01 - a11
            U[a_idx[(0,1)]][col+6] += 1
            U[a_idx[(1,1)]][col+6] += -1
            
            # Strassen V coefficients
            # v1 = b00 + b11
            V[b_idx[(0,0)]][col] += 1
            V[b_idx[(1,1)]][col] += 1
            # v2 = b00
            V[b_idx[(0,0)]][col+1] += 1
            # v3 = b01 - b11
            V[b_idx[(0,1)]][col+2] += 1
            V[b_idx[(1,1)]][col+2] += -1
            # v4 = b10 - b00
            V[b_idx[(1,0)]][col+3] += 1
            V[b_idx[(0,0)]][col+3] += -1
            # v5 = b11
            V[b_idx[(1,1)]][col+4] += 1
            # v6 = b00 + b01
            V[b_idx[(0,0)]][col+5] += 1
            V[b_idx[(0,1)]][col+5] += 1
            # v7 = b10 + b11
            V[b_idx[(1,0)]][col+6] += 1
            V[b_idx[(1,1)]][col+6] += 1
            
            # Strassen W coefficients
            # c00 = m1 + m4 - m5 + m7
            W[c_idx[(0,0)]][col] += 1
            W[c_idx[(0,0)]][col+3] += 1
            W[c_idx[(0,0)]][col+4] += -1
            W[c_idx[(0,0)]][col+6] += 1
            
            # c01 = m3 + m5
            W[c_idx[(0,1)]][col+2] += 1
            W[c_idx[(0,1)]][col+4] += 1
            
            # c10 = m2 + m4
            W[c_idx[(1,0)]][col+1] += 1
            W[c_idx[(1,0)]][col+3] += 1
            
            # c11 = m1 - m2 + m3 + m6
            W[c_idx[(1,1)]][col] += 1
            W[c_idx[(1,1)]][col+1] += -1
            W[c_idx[(1,1)]][col+2] += 1
            W[c_idx[(1,1)]][col+5] += 1
            
            col += 7
            
    return {"U": U, "V": V, "W": W}
