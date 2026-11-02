import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    The multiplication tensor is decomposed into 7 blocks of 2x2 * 2x2 multiplications.
    Each block uses Strassen's rank-7 algorithm.
    Total rank = 7 * 7 = 49.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28

    num_blocks = p // 2  # 7 blocks
    rank = num_blocks * 7  # 49

    # Strassen coefficients for 2x2 * 2x2
    # U (A), V (B), W (C)
    # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    u_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    v_strassen = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    w_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]

    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            
            # Fill U
            for a_idx in range(4):
                coeff = u_strassen[s][a_idx]
                if coeff != 0:
                    a_i = a_idx // m
                    a_j = a_idx % m
                    U[a_i * m + a_j][col] = coeff

            # Fill V
            # Block t covers B columns 2*t and 2*t+1
            for b_idx in range(4):
                coeff = v_strassen[s][b_idx]
                if coeff != 0:
                    b_j = b_idx // 2
                    b_col = b_idx % 2
                    b_k = 2 * t + b_col
                    V[b_j * p + b_k][col] = coeff

            # Fill W
            # Block t covers C columns 2*t and 2*t+1
            for c_idx in range(4):
                coeff = w_strassen[s][c_idx]
                if coeff != 0:
                    c_i = c_idx // 2
                    c_col = c_idx % 2
                    c_k = 2 * t + c_col
                    W[c_k * n + c_i][col] = coeff

    # Time check
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
