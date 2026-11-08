import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strategy: Use Strassen 2x2x2 blocks
    # Split m=4 into two groups of 2, and p=8 into four groups of 2
    # This gives 2 * 4 = 8 blocks of 2x2x2, each with rank 7
    # Total rank: 8 * 7 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen 2x2x2 decomposition
    # For 2x2 matrices: C = A * B
    # U: 4x7 (indices i*2+j), V: 4x7 (indices j*2+k), W: 4x7 (indices k*2+i)
    strassen_U = [
        [(0, 1), (3, 1)],   # (a00+a11)
        [(2, 1), (3, 1)],   # (a10+a11)
        [(0, 1)],           # a00
        [(3, 1)],           # a11
        [(0, 1), (1, 1)],   # (a00+a01)
        [(2, 1), (0, -1)],  # (a10-a00)
        [(1, 1), (3, -1)],  # (a01-a11)
    ]
    strassen_V = [
        [(0, 1), (3, 1)],   # (b00+b11)
        [(0, 1)],           # b00
        [(1, 1), (3, -1)],  # (b01-b11)
        [(2, 1), (0, -1)],  # (b10-b00)
        [(3, 1)],           # b11
        [(0, 1), (1, 1)],   # (b00+b01)
        [(2, 1), (3, 1)],   # (b10+b11)
    ]
    strassen_W = [
        [(0, 1), (3, 1)],   # (c00+c11)
        [(1, 1), (3, -1)],  # (c01-c11)
        [(2, 1), (3, 1)],   # (c10+c11)
        [(0, 1), (1, 1)],   # (c00+c01)
        [(0, -1), (2, 1)],  # (c10-c00)
        [(3, 1)],           # c11
        [(0, 1)],           # c00
    ]
    
    # Add a 2x2x2 Strassen block at given offsets
    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            cr = base_rank + r
            # U: local index = i*2+j, global = (i+i_offset)*4 + (j+j_offset)
            for idx, val in strassen_U[r]:
                li, lj = divmod(idx, 2)
                gi = li + i_offset
                gj = lj + j_offset
                U[gi * 4 + gj][cr] += val
            # V: local index = j*2+k, global = (j+j_offset)*8 + (k+k_offset)
            for idx, val in strassen_V[r]:
                lj, lk = divmod(idx, 2)
                gj = lj + j_offset
                gk = lk + k_offset
                V[gj * 8 + gk][cr] += val
            # W: local index = k*2+i, global = (k+k_offset)*2 + (i+i_offset)
            for idx, val in strassen_W[r]:
                lk, li = divmod(idx, 2)
                gk = lk + k_offset
                gi = li + i_offset
                W[gk * 2 + gi][cr] += val
    
    # 8 blocks: j_offset in {0, 2}, k_offset in {0, 2, 4, 6}
    block_idx = 0
    for j_offset in [0, 2]:
        for k_offset in [0, 2, 4, 6]:
            base_rank = block_idx * 7
            add_block(base_rank, 0, j_offset, k_offset)
            block_idx += 1
    
    # Verify with a simple test
    def verify():
        import numpy as np
        np.random.seed(42)
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C_expected = A @ B
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for r in range(rank):
                        total += float(A[i, j]) * U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                C_computed[i, k] = total
        
        return np.allclose(C_computed, C_expected, atol=1e-9)
    
    if not verify():
        # Should not happen, but fallback to naive
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
