import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 1. 2x1x7 block for A[:, 0] and B[0, :] -> 7 terms
    # 2. 3 Strassen blocks (2x2x2) for A[:, 1:3] and B[1:3, 0:6] -> 3 * 7 = 21 terms
    # 3. 1 Schoolbook block (2x2x1) for A[:, 1:3] and B[1:3, 6] -> 4 terms
    # Total rank = 7 + 21 + 4 = 32
    
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, coeffs=None):
        nonlocal col
        if coeffs is None:
            coeffs = [1] * len(u_indices)
        for idx, c in zip(u_indices, coeffs):
            U[idx][col] += c
        for idx in v_indices:
            V[idx][col] += 1
        for idx in w_indices:
            W[idx][col] += 1
        col += 1

    # Part 1: 2x1x7 block
    # A[i, 0] * B[0, k] -> C[i, k]
    # A indices: 0, 3
    # B indices: 0..6
    # W indices: k*2 + i
    for k in range(7):
        # i=0
        add_term([0], [k], [k*2 + 0])
        # i=1
        add_term([3], [k], [k*2 + 1])
        
    # Part 2: 2x2x7 block for A[:, 1:3] and B[1:3, :]
    # A local indices:
    # A00 -> A[0, 1] = 1
    # A01 -> A[0, 2] = 2
    # A10 -> A[1, 1] = 4
    # A11 -> A[1, 2] = 5
    
    # B local indices for block starting at k_start:
    # B00 -> B[1, k_start] = 7 + k_start
    # B01 -> B[1, k_start + 1] = 7 + k_start + 1
    # B10 -> B[2, k_start] = 14 + k_start
    # B11 -> B[2, k_start + 1] = 14 + k_start + 1
    
    # W local indices for block starting at k_start:
    # C00 -> W[k_start*2 + 0]
    # C01 -> W[(k_start+1)*2 + 0]
    # C10 -> W[k_start*2 + 1]
    # C11 -> W[(k_start+1)*2 + 1]
    
    def add_strassen_block(k_start):
        a00, a01, a10, a11 = 1, 2, 4, 5
        b00 = 7 + k_start
        b01 = 7 + k_start + 1
        b10 = 14 + k_start
        b11 = 14 + k_start + 1
        
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term([a00, a11], [b00, b11], [c00, c11])
        
        # M2: (A10+A11)B00 -> C10, -C11
        add_term([a10, a11], [b00], [c10, c11])
        W[c11][col-1] = -1
        
        # M3: A00(B01-B11) -> C01, C11
        add_term([a00], [b01, b11], [c01, c11])
        V[b11][col-1] = -1
        
        # M4: A11(B10-B00) -> C00, C10
        add_term([a11], [b10, b00], [c00, c10])
        V[b00][col-1] = -1
        
        # M5: (A00+A01)B11 -> -C00, C01
        add_term([a00, a01], [b11], [c00, c01])
        W[c00][col-1] = -1
        
        # M6: (A10-A00)(B00+B01) -> C11
        add_term([a10, a00], [b00, b01], [c11])
        U[a00][col-1] = -1
        
        # M7: (A01-A11)(B10+B11) -> C00
        add_term([a01, a11], [b10, b11], [c00])
        U[a11][col-1] = -1

    # Add 3 Strassen blocks for k=0,1; 2,3; 4,5
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Part 3: Schoolbook block for k=6
    # C[0, 6] = A[0, 1] B[1, 6] + A[0, 2] B[2, 6]
    # C[1, 6] = A[1, 1] B[1, 6] + A[1, 2] B[2, 6]
    
    b06 = 7 + 6
    b16 = 14 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([1], [b06], [c06])
    add_term([2], [b16], [c06])
    add_term([4], [b06], [c16])
    add_term([5], [b16], [c16])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
