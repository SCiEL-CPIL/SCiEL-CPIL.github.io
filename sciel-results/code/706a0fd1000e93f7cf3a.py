"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 terms
    # U, V, W are lists of 4 coefficients
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    # Build 4x4x4 decomposition
    # Terms are (U_vec, V_vec, W_vec) of length 16
    terms_4x4 = []
    
    for u_block, v_block, w_block in strassen_2x2:
        for u_sub, v_sub, w_sub in strassen_2x2:
            U = [0] * 16
            V = [0] * 16
            W = [0] * 16
            
            # Expand U
            for b in range(4):
                if u_block[b] != 0:
                    for l in range(4):
                        if u_sub[l] != 0:
                            idx = b * 4 + l
                            U[idx] += u_block[b] * u_sub[l]
                            
            # Expand V
            for b in range(4):
                if v_block[b] != 0:
                    for l in range(4):
                        if v_sub[l] != 0:
                            idx = b * 4 + l
                            V[idx] += v_block[b] * v_sub[l]
                            
            # Expand W
            for b in range(4):
                if w_block[b] != 0:
                    for l in range(4):
                        if w_sub[l] != 0:
                            idx = b * 4 + l
                            W[idx] += w_block[b] * w_sub[l]
                            
            terms_4x4.append((U, V, W))
            
    if time.time() > deadline:
         # Fallback to schoolbook if we run out of time (should not happen)
         rank = n * m * p
         U = [[0] * rank for _ in range(n * m)]
         V = [[0] * rank for _ in range(m * p)]
         W = [[0] * rank for _ in range(p * n)]
         column = 0
         for i in range(n):
             for j in range(m):
                 for k in range(p):
                     U[i * m + j][column] = 1
                     V[j * p + k][column] = 1
                     W[k * n + i][column] = 1
                     column += 1
         return {"U": U, "V": V, "W": W}

    # Build 4x4x8 decomposition
    # Two copies of 4x4x4
    rank = len(terms_4x4) * 2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for U4, V4, W4 in terms_4x4:
        # Left part (cols 0-3 of B and C)
        for i in range(16):
            if U4[i] != 0:
                U[i][col] = U4[i]
                
        for i in range(16):
            if V4[i] != 0:
                # Map V index j*4+k -> j*8+k
                j = i // 4
                k = i % 4
                full_idx = j * 8 + k
                V[full_idx][col] = V4[i]
                
        for i in range(16):
            if W4[i] != 0:
                # Map W index k*4+i -> k*4+i (no change for left part)
                W[i][col] = W4[i]
                
        col += 1
        
        # Right part (cols 4-7 of B and C)
        for i in range(16):
            if U4[i] != 0:
                U[i][col] = U4[i]
                
        for i in range(16):
            if V4[i] != 0:
                # Map V index j*4+k -> j*8+4+k
                j = i // 4
                k = i % 4
                full_idx = j * 8 + 4 + k
                V[full_idx][col] = V4[i]
                
        for i in range(16):
            if W4[i] != 0:
                # Map W index k*4+i -> (k+4)*4+i = k*4+i+16
                full_idx = i + 16
                W[full_idx][col] = W4[i]
                
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
