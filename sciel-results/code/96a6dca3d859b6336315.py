"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 21 decomposition using 3 Strassen blocks
    rank = 21
    
    # U shape: (n*m, rank) = (4, 21)
    # V shape: (m*p, rank) = (12, 21)
    # W shape: (p*n, rank) = (12, 21)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2x2
    # U indices: 00, 01, 10, 11
    u_strassen = [
        [1, 0, 0, 1], # M1: a00 + a11
        [0, 0, 1, 1], # M2: a10 + a11
        [1, 0, 0, 0], # M3: a00
        [0, 0, 0, 1], # M4: a11
        [1, 1, 0, 0], # M5: a00 + a01
        [0, 0, 1, -1], # M6: a10 - a00
        [0, 1, 0, -1]  # M7: a01 - a11
    ]
    
    # V indices: 00, 01, 10, 11 (local j, k')
    v_strassen = [
        [1, 0, 0, 1], # M1: b00 + b11
        [1, 0, 0, 0], # M2: b00
        [0, 1, 0, -1], # M3: b01 - b11
        [0, 0, 1, -1], # M4: b10 - b00
        [0, 0, 0, 1], # M5: b11
        [1, 1, 0, 0], # M6: b00 + b01
        [0, 0, 1, 1]  # M7: b10 + b11
    ]
    
    # W indices: 00, 10, 01, 11 (local k', i)
    # c00 (k'=0,i=0), c10 (k'=0,i=1), c01 (k'=1,i=0), c11 (k'=1,i=1)
    w_strassen = [
        [1, 0, 0, 1], # M1: c00 + c11
        [0, 1, 0, -1], # M2: c10 - c11
        [0, 0, 1, 1], # M3: c01 + c11
        [1, 1, 0, 0], # M4: c00 + c10
        [-1, 0, 1, 0], # M5: -c00 + c01
        [0, 0, 0, 1], # M6: c11
        [1, 0, 0, 0]  # M7: c00
    ]
    
    # 3 blocks for p=6
    num_blocks = 3
    block_size = 2
    
    for g in range(num_blocks):
        for r in range(7):
            col = g * 7 + r
            
            # Fill U
            for i in range(4):
                U[i][col] = u_strassen[r][i]
                
            # Fill V
            # Local index l corresponds to j, k'
            # Global index is j * p + (g * block_size + k')
            for l in range(4):
                j = l // 2
                kp = l % 2
                global_k = g * block_size + kp
                global_idx = j * p + global_k
                V[global_idx][col] = v_strassen[r][l]
                
            # Fill W
            # Local index l corresponds to k', i
            # Global index is (g * block_size + k') * n + i
            for l in range(4):
                kp = l // 2
                i = l % 2
                global_k = g * block_size + kp
                global_idx = global_k * n + i
                W[global_idx][col] = w_strassen[r][l]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
