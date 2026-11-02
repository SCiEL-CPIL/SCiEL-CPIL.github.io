import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    timeout = 145  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # Block decomposition: split p=9 into 3 blocks of size p_block=3
    # Each block computes a 2x3 * 3x3 -> 2x3 multiplication
    # Schoolbook rank per block: n * m * p_block = 2*3*3 = 18
    # Total rank: 3 * 18 = 54 (same as full schoolbook, but structured)
    
    p_block = 3
    num_blocks = p // p_block  # 3
    rank_per_block = n * m * p_block  # 18
    rank = num_blocks * rank_per_block  # 54
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for block in range(num_blocks):
        # For each block, do schoolbook decomposition of 2x3 * 3x3
        for i in range(n):
            for j in range(m):
                for k_block in range(p_block):
                    k = block * p_block + k_block  # global k index
                    # U index: i * m + j
                    U[i * m + j, col] = 1
                    # V index: j * p + k
                    V[j * p + k, col] = 1
                    # W index: k * n + i (as per contract: W[k*n+i, r])
                    W[k * n + i, col] = 1
                    col += 1
    
    # Check deadline
    if time.time() - start_time > timeout - 5:
        pass  # Return immediately
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
