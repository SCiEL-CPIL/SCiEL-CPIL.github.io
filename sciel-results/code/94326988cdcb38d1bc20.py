import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, safety margin for 120s/150s
    
    # Return the known R=39 solution
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, signs=None):
        nonlocal col
        if signs is None:
            signs = [1] * len(u_indices)
        for idx, s in zip(u_indices, signs):
            U[idx][col] += s
        for idx in v_indices:
            V[idx][col] += 1
        for idx in w_indices:
            W[idx][col] += 1
        col += 1

    def add_strassen_block(k_start):
        # A indices (2x2 part of 2x3 A):
        # A00: 0
        # A01: 1
        # A10: 3
        # A11: 4
        
        # B indices (2x2 part of 3x7 B, rows 0-1, cols k_start, k_start+1):
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        # C indices (2x2 part of 2x7 C, rows 0-1, cols k_start, k_start+1):
        # C index = k*2 + i (col * n + row)
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term([0, 4], [b00, b11], [c00, c11])
        
        # M2: (A10+A11)B00 -> C10, -C11
        add_term([3, 4], [b00], [c10, c11])
        W[c11][col-1] = -1
        
        # M3: A00(B01-B11) -> C01, C11
        add_term([0], [b01, b11], [c01, c11])
        V[b11][col-1] = -1
        
        # M4: A11(B10-B00) -> C00, C10
        add_term([4], [b10, b00], [c00, c10])
        V[b00][col-1] = -1
        
        # M5: (A00+A01)B11 -> -C00, C01
        add_term([0, 1], [b11], [c00, c01])
        W[c00][col-1] = -1
        
        # M6: (A10-A00)(B00+B01) -> C11
        add_term([3, 0], [b00, b01], [c11])
        U[0][col-1] = -1
        
        # M7: (A01-A11)(B10+B11) -> C00
        add_term([1, 4], [b10, b11], [c00])
        U[4][col-1] = -1

    # 3 Strassen blocks for cols 0-1, 2-3, 4-5
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Block for col 6 of B (rows 0-1)
    # A00 B06 -> C06
    # A01 B16 -> C06
    # A10 B06 -> C16
    # A11 B16 -> C16
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])
    
    # Part for A[:, 2] * B[2, :]
    # A02 B2k -> C0k
    # A12 B2k -> C1k
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])
        
    # Check deadline (should not be reached)
    if time.time() > deadline:
        pass # Return best found
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
