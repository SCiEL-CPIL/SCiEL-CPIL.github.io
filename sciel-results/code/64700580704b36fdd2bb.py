"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 2-level Strassen decomposition
    # Rank = 7 * 7 = 49
    
    # Define 2x2 Strassen coefficients for the "inner" level
    # S_A[l][local_idx] where local_idx = r*2 + c for 2x2 block
    S_A = [
        [1, 0, 0, 1], # l=0: A00 + A11
        [0, 0, 1, 1], # l=1: A10 + A11
        [1, 0, 0, 0], # l=2: A00
        [0, 0, 0, 1], # l=3: A11
        [1, 1, 0, 0], # l=4: A00 + A01
        [-1, 0, 1, 0],# l=5: A10 - A00
        [0, 1, 0, -1] # l=6: A01 - A11
    ]
    
    S_B = [
        [1, 0, 0, 1], # l=0: B00 + B11
        [1, 0, 0, 0], # l=1: B00
        [0, 1, 0, -1],# l=2: B01 - B11
        [-1, 0, 1, 0],# l=3: B10 - B00
        [0, 0, 0, 1], # l=4: B11
        [1, 1, 0, 0], # l=5: B00 + B01
        [0, 0, 1, 1]  # l=6: B10 + B11
    ]
    
    S_C = [
        [1, 0, 0, 1], # l=0: C00 + C11
        [0, 0, 1, -1],# l=1: C10 - C11
        [0, 1, 0, 1], # l=2: C01 + C11
        [1, 0, 1, 0], # l=3: C00 + C10
        [-1, 1, 0, 0],# l=4: -C00 + C01
        [0, 0, 0, 1], # l=5: C11
        [1, 0, 0, 0]  # l=6: C00
    ]
    
    # For the "outer" level, we use the same Strassen patterns
    # OA[K][block_idx] where block_idx = r*2 + c for 2x2 block of 2x2 matrices
    OA = S_A
    OB = S_B
    OC = S_C
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for K in range(7):
        for l in range(7):
            r = K * 7 + l
            
            # Fill U: U[i*m + j, r]
            for i in range(n):
                for j in range(m):
                    rb = i // 2  # row block
                    cb = j // 2  # col block
                    li = i % 2   # local row
                    lj = j % 2   # local col
                    block_idx = rb * 2 + cb
                    local_idx = li * 2 + lj
                    coeff = OA[K][block_idx] * S_A[l][local_idx]
                    U[i * m + j][r] = coeff
                        
            # Fill V: V[j*p + k, r]
            for j in range(m):
                for k in range(p):
                    rb = j // 2
                    cb = k // 2
                    li = j % 2
                    lj = k % 2
                    block_idx = rb * 2 + cb
                    local_idx = li * 2 + lj
                    coeff = OB[K][block_idx] * S_B[l][local_idx]
                    V[j * p + k][r] = coeff
                        
            # Fill W: W[k*n + i, r]
            for k in range(p):
                for i in range(n):
                    rb = i // 2
                    cb = k // 2
                    li = i % 2
                    lk = k % 2
                    block_idx = rb * 2 + cb
                    local_idx = li * 2 + lk
                    coeff = OC[K][block_idx] * S_C[l][local_idx]
                    W[k * n + i][r] = coeff
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
