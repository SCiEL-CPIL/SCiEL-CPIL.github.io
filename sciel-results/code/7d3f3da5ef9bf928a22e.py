import time
import sys

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    n, m, p = DIMENSIONS
    
    # Base 2x2x2 Strassen decomposition
    # U indices: i*2+j (0..3)
    # V indices: j*2+k (0..3)
    # W indices: k*2+i (0..3)
    
    U_base = [
        [1, 1, 1, 0, 1, -1, 1], # i=0, j=0 (a)
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1 (b)
        [0, 1, 0, 0, 0, 1, -1], # i=1, j=0 (c)
        [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1 (d)
    ]
    
    V_base = [
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0 (e)
        [0, 0, 1, 0, 0, 1, 0],  # j=0, k=1 (f)
        [0, 0, 0, 1, 0, 0, 1],  # j=1, k=0 (g)
        [1, 0, -1, 0, 1, 0, 1]  # j=1, k=1 (h)
    ]
    
    W_base = [
        [1, 0, 0, -1, 1, 0, -1], # k=0, i=0 (C00)
        [0, 1, 0, 0, 0, 0, 1],   # k=0, i=1 (C10)
        [0, 0, 1, 1, 0, 0, 0],   # k=1, i=0 (C01)
        [1, -1, 1, 0, 0, 1, 0]   # k=1, i=1 (C11)
    ]
    
    rank_base = 7
    
    # Construct 2x2x8 decomposition
    # n=2, m=2, p=8
    rank_228 = 4 * rank_base # 28
    U_228 = [[0] * rank_228 for _ in range(4)]
    V_228 = [[0] * rank_228 for _ in range(16)]
    W_228 = [[0] * rank_228 for _ in range(16)]
    
    for l in range(4):
        offset = l * rank_base
        # U is same for all blocks
        for i in range(2):
            for j in range(2):
                u_idx = i * 2 + j
                for r in range(rank_base):
                    U_228[u_idx][offset + r] = U_base[u_idx][r]
        
        # V and W depend on block l
        for j in range(2):
            for k_prime in range(2):
                k = l * 2 + k_prime
                v_idx = j * 8 + k
                for r in range(rank_base):
                    V_228[v_idx][offset + r] = V_base[j * 2 + k_prime][r]
        
        for i in range(2):
            for k_prime in range(2):
                k = l * 2 + k_prime
                w_idx = k * 2 + i
                for r in range(rank_base):
                    W_228[w_idx][offset + r] = W_base[k_prime * 2 + i][r]

    # Construct 2x4x8 decomposition
    # n=2, m=4, p=8
    rank_total = 2 * rank_228 # 56
    U = [[0] * rank_total for _ in range(8)]
    V = [[0] * rank_total for _ in range(32)]
    W = [[0] * rank_total for _ in range(16)]
    
    # Block 0: j in 0,1
    for i in range(2):
        for j_prime in range(2):
            j = j_prime
            u_idx = i * 4 + j
            u_228_idx = i * 2 + j_prime
            for r in range(rank_228):
                U[u_idx][r] = U_228[u_228_idx][r]
        
        for k in range(8):
            v_idx = j * 8 + k
            v_228_idx = j_prime * 8 + k
            for r in range(rank_228):
                V[v_idx][r] = V_228[v_228_idx][r]
    
    # Block 1: j in 2,3
    offset_2 = rank_228
    for i in range(2):
        for j_prime in range(2):
            j = 2 + j_prime
            u_idx = i * 4 + j
            u_228_idx = i * 2 + j_prime
            for r in range(rank_228):
                U[u_idx][offset_2 + r] = U_228[u_228_idx][r]
        
        for k in range(8):
            v_idx = j * 8 + k
            v_228_idx = j_prime * 8 + k
            for r in range(rank_228):
                V[v_idx][offset_2 + r] = V_228[v_228_idx][r]
    
    # W is sum of both blocks (same indices)
    for i in range(2):
        for k in range(8):
            w_idx = k * 2 + i
            w_228_idx = k * 2 + i
            for r in range(rank_228):
                W[w_idx][r] = W_228[w_228_idx][r]
                W[w_idx][offset_2 + r] = W_228[w_228_idx][r]

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
