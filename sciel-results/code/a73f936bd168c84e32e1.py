import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We return the known valid decomposition of rank 39.
    # Structure:
    # - 3 Strassen blocks (2x2x2) for k pairs (0,1), (2,3), (4,5) -> Rank 21
    # - 1 Schoolbook block (2x2x1) for k=6 -> Rank 4
    # - 1 Schoolbook block (2x1x7) for j=2 -> Rank 14
    # Total Rank = 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += 1
        for idx in v_indices:
            V[idx][col] += 1
        for idx in w_indices:
            W[idx][col] += 1
        col += 1

    def add_strassen_block(k_start):
        # k_start is 0, 2, 4
        # A indices (i*3+j):
        # A00: 0, A01: 1, A10: 3, A11: 4
        # B indices (j*7+k):
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        # C indices (k*2+i):
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

    # Add 3 Strassen blocks for k=0, 2, 4
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Block for k=6 (2x2x1 Schoolbook)
    # C06 = A00 B06 + A01 B16
    # C16 = A10 B06 + A11 B16
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])
    
    # Block for j=2 (2x1x7 Schoolbook)
    # C0k = A02 B2k
    # C1k = A12 B2k
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])
        
    # Verification step (optional but safe)
    # We trust the construction, but we can quickly check dimensions
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        raise ValueError("Dimension mismatch")
    for row in U:
        if len(row) != rank: raise ValueError("U width mismatch")
    for row in V:
        if len(row) != rank: raise ValueError("V width mismatch")
    for row in W:
        if len(row) != rank: raise ValueError("W width mismatch")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
