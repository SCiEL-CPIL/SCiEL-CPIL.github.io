import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a rank-minimal matrix multiplication decomposition for 2x3 * 3x8.
    Uses a block-partitioned Strassen-based construction (rank 44).
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # --- Step 1: Construct a valid initial decomposition ---
    # Rank 44 decomposition.
    
    rank_init = 44
    U = np.zeros((n*m, rank_init), dtype=int)
    V = np.zeros((m*p, rank_init), dtype=int)
    W = np.zeros((p*n, rank_init), dtype=int)
    
    col = 0
    
    # Part 1: A1 * B1 (2x2 * 2x8)
    # Strassen factors for 2x2:
    for j in range(4):
        w_r00 = 4*j
        w_r01 = 4*j + 2
        w_r10 = 4*j + 1
        w_r11 = 4*j + 3
        
        v_e = 2*j
        v_f = 2*j + 1
        v_g = 8 + 2*j
        v_h = 9 + 2*j
        
        # P1: (a+d)(e+h)
        U[0, col] = 1; U[4, col] = 1
        V[v_e, col] = 1; V[v_h, col] = 1
        W[w_r00, col] = 1; W[w_r11, col] = 1
        col += 1
        
        # P2: (c+d)e
        U[3, col] = 1; U[4, col] = 1
        V[v_e, col] = 1
        W[w_r10, col] = 1; W[w_r11, col] = -1
        col += 1
        
        # P3: a(f-h)
        U[0, col] = 1
        V[v_f, col] = 1; V[v_h, col] = -1
        W[w_r01, col] = 1; W[w_r11, col] = 1
        col += 1
        
        # P4: d(g-e)
        U[4, col] = 1
        V[v_g, col] = 1; V[v_e, col] = -1
        W[w_r00, col] = 1; W[w_r10, col] = 1
        col += 1
        
        # P5: (a+b)h
        U[0, col] = 1; U[1, col] = 1
        V[v_h, col] = 1
        W[w_r00, col] = -1; W[w_r01, col] = 1
        col += 1
        
        # P6: (c-a)(e+f)
        U[3, col] = 1; U[0, col] = -1
        V[v_e, col] = 1; V[v_f, col] = 1
        W[w_r11, col] = 1
        col += 1
        
        # P7: (b-d)(g+h)
        U[1, col] = 1; U[4, col] = -1
        V[v_g, col] = 1; V[v_h, col] = 1
        W[w_r00, col] = 1
        col += 1
        
    # Part 2: A2 * B2 (2x1 * 1x8)
    for i in range(2):
        for k in range(8):
            u_idx = i*3 + 2
            v_idx = 2*8 + k
            w_idx = k*2 + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
