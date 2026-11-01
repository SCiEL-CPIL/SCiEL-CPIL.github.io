import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    timeout = 120.0
    
    # Strassen-based decomposition: partition p=12 columns into 6 pairs
    # Each pair uses 7 rank-1 terms (Strassen for 2x2x2 matrix mult)
    # Total rank = 6 * 7 = 42
    
    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 42)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # shape (24, 42)
    
    # Strassen 7-product decomposition for 2x2x2
    strassen_terms = [
        {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1},       # M1
        {2: 1, 3: 1}, {0: 1},        {2: 1, 3: -1},      # M2
        {0: 1},       {1: 1, 3: -1}, {1: 1, 3: 1},       # M3
        {3: 1},       {2: 1, 0: -1}, {0: 1, 2: 1},       # M4
        {0: 1, 1: 1}, {3: 1},        {0: -1, 1: 1},      # M5
        {0: -1, 2: 1},{0: 1, 1: 1},  {3: 1},             # M6
        {1: 1, 3: -1},{2: 1, 3: 1},  {0: 1},             # M7
    ]
    
    for t in range(num_blocks):
        if time.time() - start_time > timeout - 5:
            break
        for s in range(7):
            col = t * 7 + s
            u_coeffs = strassen_terms[s * 3]
            v_coeffs = strassen_terms[s * 3 + 1]
            w_coeffs = strassen_terms[s * 3 + 2]
            
            # Set U
            for a_idx, val in u_coeffs.items():
                U[a_idx][col] = val
            
            # Set V
            for local_b, val in v_coeffs.items():
                if local_b == 0:
                    global_b = 2 * t
                elif local_b == 1:
                    global_b = 2 * t + 1
                elif local_b == 2:
                    global_b = m * p // 2 + 2 * t
                else:
                    global_b = m * p // 2 + 2 * t + 1
                V[global_b][col] = val
            
            # Set W
            for local_c, val in w_coeffs.items():
                if local_c == 0:
                    global_w = (2 * t) * n + 0
                elif local_c == 1:
                    global_w = (2 * t + 1) * n + 0
                elif local_c == 2:
                    global_w = (2 * t) * n + 1
                else:
                    global_w = (2 * t + 1) * n + 1
                W[global_w][col] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
