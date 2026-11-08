import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 110.0  # Safety margin before 120s limit

    # 2-level Strassen decomposition
    # Rank = 7 * 7 = 49
    
    # Define 2x2 Strassen coefficients
    # S_A[l][local_idx] where local_idx = r*2 + c
    S_A = np.array([
        [1, 0, 0, 1], # l=0: A00 + A11
        [0, 0, 1, 1], # l=1: A10 + A11
        [1, 0, 0, 0], # l=2: A00
        [0, 0, 0, 1], # l=3: A11
        [1, 1, 0, 0], # l=4: A00 + A01
        [-1, 0, 1, 0],# l=5: A10 - A00
        [0, 1, 0, -1] # l=6: A01 - A11
    ], dtype=np.int64)
    
    S_B = np.array([
        [1, 0, 0, 1], # l=0: B00 + B11
        [1, 0, 0, 0], # l=1: B00
        [0, 1, 0, -1],# l=2: B01 - B11
        [-1, 0, 1, 0],# l=3: B10 - B00
        [0, 0, 0, 1], # l=4: B11
        [1, 1, 0, 0], # l=5: B00 + B01
        [0, 0, 1, 1]  # l=6: B10 + B11
    ], dtype=np.int64)
    
    S_C = np.array([
        [1, 0, 0, 1], # l=0: C00 + C11
        [0, 0, 1, -1],# l=1: C10 - C11
        [0, 1, 0, 1], # l=2: C01 + C11
        [1, 0, 1, 0], # l=3: C00 + C10
        [-1, 1, 0, 0],# l=4: -C00 + C01
        [0, 0, 0, 1], # l=5: C11
        [1, 0, 0, 0]  # l=6: C00
    ], dtype=np.int64)
    
    # Define 4x4 (2x2 blocks of 2x2) Strassen coefficients
    # OA[K][block_idx] where block_idx = r*2 + c
    OA = S_A.copy() # Same structure for outer level
    OB = S_B.copy()
    OC = S_C.copy()
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Pre-compute indices for efficiency
    # i, j for U (n x m)
    # j, k for V (m x p)
    # k, i for W (p x n)
    
    for K in range(7):
        for l in range(7):
            r = K * 7 + l
            
            if time.time() - start_time > timeout:
                # Return partial result if timeout approaches
                # In this case, the loop is fast enough, but we guard just in case
                break
                
            # Fill U
            # U[i*m + j, r] = OA[K][i//2 * 2 + j//2] * S_A[l][i%2 * 2 + j%2]
            for i in range(n):
                for j in range(m):
                    rb = i // 2
                    cb = j // 2
                    li = i % 2
                    lj = j % 2
                    block_idx = rb * 2 + cb
                    local_idx = li * 2 + lj
                    coeff = OA[K][block_idx] * S_A[l][local_idx]
                    if coeff != 0:
                        U[i * m + j][r] = coeff
                        
            # Fill V
            # V[j*p + k, r] = OB[K][j//2 * 2 + k//2] * S_B[l][j%2 * 2 + k%2]
            for j in range(m):
                for k in range(p):
                    rb = j // 2
                    cb = k // 2
                    li = j % 2
                    lj = k % 2
                    block_idx = rb * 2 + cb
                    local_idx = li * 2 + lj
                    coeff = OB[K][block_idx] * S_B[l][local_idx]
                    if coeff != 0:
                        V[j * p + k][r] = coeff
                        
            # Fill W
            # W[k*n + i, r] = OC[K][i//2 * 2 + k//2] * S_C[l][i%2 * 2 + k%2]
            for k in range(p):
                for i in range(n):
                    # C[i, k]
                    rb = i // 2
                    cb = k // 2
                    li = i % 2
                    lk = k % 2
                    block_idx = rb * 2 + cb
                    local_idx = li * 2 + lk
                    coeff = OC[K][block_idx] * S_C[l][local_idx]
                    if coeff != 0:
                        W[k * n + i][r] = coeff
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
