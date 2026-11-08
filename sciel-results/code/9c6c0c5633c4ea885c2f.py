import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 3 Strassen blocks (2x2x2) for k=0,1; 2,3; 4,5 -> 3 * 7 = 21
    # 1 Schoolbook block (2x2x1) for k=6 -> 4
    # 1 Schoolbook block (2x1x7) for j=2 -> 14
    rank = 21 + 4 + 14
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    start_time = time.time()
    deadline = start_time + 140.0
    
    def add_term(u_indices, v_indices, w_indices, signs_u=None, signs_v=None, signs_w=None):
        nonlocal col
        if time.time() > deadline:
            return
        if signs_u is None:
            signs_u = [1] * len(u_indices)
        if signs_v is None:
            signs_v = [1] * len(v_indices)
        if signs_w is None:
            signs_w = [1] * len(w_indices)
        
        for idx, sign in zip(u_indices, signs_u):
            U[idx][col] += sign
        for idx, sign in zip(v_indices, signs_v):
            V[idx][col] += sign
        for idx, sign in zip(w_indices, signs_w):
            W[idx][col] += sign
        col += 1

    def add_strassen_block(k_start):
        a00 = 0 * 3 + 0
        a01 = 0 * 3 + 1
        a10 = 1 * 3 + 0
        a11 = 1 * 3 + 1
        
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1 = (A00 + A11)(B00 + B11) -> C00, C11
        add_term([a00, a11], [b00, b11], [c00, c11])
        
        # M2 = (A10 + A11)(B00) -> C10, -C11
        add_term([a10, a11], [b00], [c10, c11], signs_w=[1, -1])
        
        # M3 = (A00)(B01 - B11) -> C01, C11
        add_term([a00], [b01, b11], [c01, c11], signs_v=[1, -1])
        
        # M4 = (A11)(B10 - B00) -> C00, C10
        add_term([a11], [b10, b00], [c00, c10], signs_v=[1, -1])
        
        # M5 = (A00 + A01)(B11) -> -C00, C01
        add_term([a00, a01], [b11], [c00, c01], signs_w=[-1, 1])
        
        # M6 = (A10 - A00)(B00 + B01) -> C11
        add_term([a10, a00], [b00, b01], [c11], signs_u=[1, -1])
        
        # M7 = (A01 - A11)(B10 + B11) -> C00
        add_term([a01, a11], [b10, b11], [c00], signs_u=[1, -1])

    # Add 3 Strassen blocks
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Block for k=6, 2x2x1 Schoolbook
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])
    
    # Part 2: j=2
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
