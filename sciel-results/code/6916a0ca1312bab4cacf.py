"""Matrix multiplication 4x4x4 with rank-49 Strassen decomposition."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 exact tensor decomposition for 4x4 matrix multiplication
    using recursive Strassen's algorithm.
    """
    import time
    start_time = time.time()
    deadline = 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # 2x2 Strassen decomposition coefficients
    # U: coefficients for A elements [A00, A01, A10, A11]
    # V: coefficients for B elements [B00, B01, B10, B11]
    # W: coefficients for C elements [C00, C01, C10, C11]
    strassen_2x2_U = [
        [1, 0, 0, 1],   # M1: (A00+A11)
        [0, 0, 1, 1],   # M2: (A10+A11)
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: (A00+A01)
        [0, 0, 1, -1],  # M6: (A10-A00)
        [0, 1, 0, -1],  # M7: (A01-A11)
    ]
    strassen_2x2_V = [
        [1, 0, 0, 1],   # M1: (B00+B11)
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: (B01-B11)
        [0, 0, 1, -1],  # M4: (B10-B00)
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: (B00+B01)
        [0, 0, 1, 1],   # M7: (B10+B11)
    ]
    strassen_2x2_W = [
        [1, 0, 0, 1],   # M1: C00+C11
        [0, 0, 1, -1],  # M2: C10-C11
        [0, 1, 0, 1],   # M3: C01+C11
        [1, 0, 1, 0],   # M4: C00+C10
        [-1, 1, 0, 0],  # M5: -C00+C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, -1],  # M7: C00-C11
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            # Fill U
            for bi in range(2):
                for bj in range(2):
                    c_outer = strassen_2x2_U[t][bi * 2 + bj]
                    if c_outer == 0:
                        continue
                    for di in range(2):
                        for dj in range(2):
                            c_inner = strassen_2x2_U[s][di * 2 + dj]
                            if c_inner == 0:
                                continue
                            i = 2 * bi + di
                            j = 2 * bj + dj
                            idx = i * 4 + j
                            U[idx][col] += c_outer * c_inner
            # Fill V
            for bi in range(2):
                for bj in range(2):
                    c_outer = strassen_2x2_V[t][bi * 2 + bj]
                    if c_outer == 0:
                        continue
                    for di in range(2):
                        for dj in range(2):
                            c_inner = strassen_2x2_V[s][di * 2 + dj]
                            if c_inner == 0:
                                continue
                            i = 2 * bi + di
                            j = 2 * bj + dj
                            idx = i * 4 + j
                            V[idx][col] += c_outer * c_inner
            # Fill W
            for bi in range(2):
                for bj in range(2):
                    c_outer = strassen_2x2_W[t][bi * 2 + bj]
                    if c_outer == 0:
                        continue
                    for di in range(2):
                        for dj in range(2):
                            c_inner = strassen_2x2_W[s][di * 2 + dj]
                            if c_inner == 0:
                                continue
                            i = 2 * bi + di
                            j = 2 * bj + dj
                            idx = i * 4 + j
                            W[idx][col] += c_outer * c_inner
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
