import numpy as np
import time

DIMENSIONS = [2, 2, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s
    
    n, m, p = DIMENSIONS
    
    # Strassen parameters for 2x2x2
    # U, V, W vectors for the 7 terms
    # Indices for A: 0:00, 1:01, 2:10, 3:11
    # Indices for B: 0:00, 1:01, 2:10, 3:11
    # Indices for C: 0:00, 1:01, 2:10, 3:11
    
    U_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    
    V_strassen = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [-1, 0, 1, 0],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    W_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, -1],# P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0],# P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Pre-compute index mappings for efficiency
    v_indices_map = [[2*t, 2*t+1, 8+2*t, 8+2*t+1] for t in range(num_blocks)]
    w_indices_map = [[4*t, 4*t+2, 4*t+1, 4*t+3] for t in range(num_blocks)]
    
    col = 0
    for t in range(num_blocks):
        if time.time() > deadline:
            break
        v_indices = v_indices_map[t]
        w_indices = w_indices_map[t]
        
        for r in range(7):
            # Fill U
            for i in range(n * m):
                U[i][col] = U_strassen[r][i]
                
            # Fill V
            for l in range(4):
                V[v_indices[l]][col] = V_strassen[r][l]
                
            # Fill W
            for l in range(4):
                W[w_indices[l]][col] = W_strassen[r][l]
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
