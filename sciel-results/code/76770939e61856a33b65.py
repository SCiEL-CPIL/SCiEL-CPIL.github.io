"""Matrix multiplication 3,4,8 using rank-88 decomposition."""
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i=0, schoolbook for 1x4x8 (32 terms)
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Check deadline
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Part 2: i=1,2 using Strassen for 2x4x8 (56 terms)
    # Split j into 2 blocks of 2: [0,1] and [2,3]
    # Split k into 4 blocks of 2: [0,1], [2,3], [4,5], [6,7]
    
    # Strassen 2x2x2 factors
    # Local indices: A: 00, 01, 10, 11 -> 0, 1, 2, 3
    #                B: 00, 01, 10, 11 -> 0, 1, 2, 3
    #                C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    strassen_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ]
    
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                # U updates: local l -> global i=1+l//2, j=j_start+l%2
                for l in range(4):
                    val = strassen_U[t][l]
                    if val != 0:
                        local_i = l // 2
                        local_j = l % 2
                        g_i = 1 + local_i
                        g_j = j_start + local_j
                        U[g_i * m + g_j][col] += val
                
                # V updates: local l -> global j=j_start+l//2, k=k_start+l%2
                for l in range(4):
                    val = strassen_V[t][l]
                    if val != 0:
                        local_j = l // 2
                        local_k = l % 2
                        g_j = j_start + local_j
                        g_k = k_start + local_k
                        V[g_j * p + g_k][col] += val
                
                # W updates: local l -> global k=k_start+l%2, i=1+l//2
                for l in range(4):
                    val = strassen_W[t][l]
                    if val != 0:
                        local_i = l // 2
                        local_k = l % 2
                        g_i = 1 + local_i
                        g_k = k_start + local_k
                        W[g_k * n + g_i][col] += val
                
                col += 1
        
        # Check deadline between j_blocks
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
