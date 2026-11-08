import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The current best known simple decomposition for 2x3 by 3x7 is rank 39.
    # This is composed of:
    # 1. Rank 14 for the 2x1x7 part (j=2)
    # 2. Rank 25 for the 2x2x7 part (j=0,1)
    #    Which is 3 Strassen blocks (2x2x2) for k=0,1; 2,3; 4,5 -> 3 * 7 = 21
    #    Plus 1 Schoolbook block (2x2x1) for k=6 -> 4
    # Total: 14 + 21 + 4 = 39.
    
    rank = 39
    N_A = n * m  # 6
    N_B = m * p  # 21
    N_C = p * n  # 14
    
    U = [[0] * rank for _ in range(N_A)]
    V = [[0] * rank for _ in range(N_B)]
    W = [[0] * rank for _ in range(N_C)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, coeffs_u=None, coeffs_v=None, coeffs_w=None):
        nonlocal col
        if coeffs_u is None: coeffs_u = [1]*len(u_indices)
        if coeffs_v is None: coeffs_v = [1]*len(v_indices)
        if coeffs_w is None: coeffs_w = [1]*len(w_indices)
        
        for idx, c in zip(u_indices, coeffs_u):
            U[idx][col] += c
        for idx, c in zip(v_indices, coeffs_v):
            V[idx][col] += c
        for idx, c in zip(w_indices, coeffs_w):
            W[idx][col] += c
        col += 1

    def add_strassen_block(k_start):
        # A indices: i*3+j. i in 0,1; j in 0,1.
        # A00=0, A01=1, A10=3, A11=4
        # B indices: j*7+k. j in 0,1; k in k_start, k_start+1.
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        # C indices: k*2+i. k in k_start, k_start+1; i in 0,1.
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

    # Add 3 Strassen blocks for k=0,1; 2,3; 4,5
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Block 4: k=6, 2x2x1 Schoolbook
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
    
    # Part 2: j=2 (2x1x7 Schoolbook)
    # A02 B2k C0k
    # A12 B2k C1k
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
