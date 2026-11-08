import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    # Rank 49 for 4x4x4 (Strassen) + Rank 48 for 4x4x3 (Schoolbook) = 97
    
    rank_strassen = 49
    rank_schoolbook = 48
    total_rank = rank_strassen + rank_schoolbook
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Correct Strassen 2x2 tensors (0-indexed)
    # t=0: M1 = (A00+A11)(B00+B11)
    # t=1: M2 = (A10+A11)B00
    # t=2: M3 = A00(B01-B11)
    # t=3: M4 = A11(B10-B00)
    # t=4: M5 = (A00+A01)B11
    # t=5: M6 = (A10-A00)(B00+B01)
    # t=6: M7 = (A01-A11)(B10+B11)
    
    strassen_2x2 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1})
    ]
    
    # The block-level Strassen is the same structure
    strassen_4x4_blocks = strassen_2x2
    
    col = 0
    for s in range(7):
        U_block, V_block, W_block = strassen_4x4_blocks[s]
        for t in range(7):
            if time.time() > deadline:
                break
            U_elem, V_elem, W_elem = strassen_2x2[t]
            
            # Combine block and element levels
            for (I, J), cU in U_block.items():
                for (i, j), dU in U_elem.items():
                    gi = 2 * I + i
                    gj = 2 * J + j
                    U[gi * m + gj, col] += cU * dU
            
            for (J, K), cV in V_block.items():
                for (j, k), dV in V_elem.items():
                    gj = 2 * J + j
                    gk = 2 * K + k
                    V[gj * 4 + gk, col] += cV * dV  # p=4 for this part
            
            for (K, I), cW in W_block.items():
                for (k, i), dW in W_elem.items():
                    gk = 2 * K + k
                    gi = 2 * I + i
                    W[gk * n + gi, col] += cW * dW
            
            col += 1
    
    # 2. Schoolbook for 4x4x3 part (k in 4..6)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                if time.time() > deadline:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
