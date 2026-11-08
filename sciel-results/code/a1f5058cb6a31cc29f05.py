import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x3 by 3x7 matrix multiplication.
    
    Strategy:
    - Split the 3 columns of A into a 2x2 block (j=0,1) and a 1x1 block (j=2).
    - For the 2x2 block (j=0,1), split the 7 rows of B into three 2x2 blocks (k=0,1; k=2,3; k=4,5)
      and one 2x1 block (k=6).
    - Use Strassen's algorithm (rank 7) for each 2x2x2 block.
    - Use schoolbook (rank 4) for the 2x2x1 block.
    - Use schoolbook (rank 14) for the 2x1x7 block (j=2).
    
    Total rank = 3*7 + 4 + 14 = 21 + 4 + 14 = 39.
    """
    n, m, p = 2, 3, 7
    
    rank = 3 * 7 + 4 + 14  # 39
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 39
    V = [[0] * rank for _ in range(m * p)]  # 21 x 39
    W = [[0] * rank for _ in range(p * n)]  # 14 x 39
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, signs_u=None, signs_v=None, signs_w=None):
        nonlocal col
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
        """
        Add Strassen decomposition for a 2x2x2 block.
        A is 2x2: indices (i, j) where i in {0,1}, j in {0,1}
        B is 2x2: indices (j, k) where j in {0,1}, k in {k_start, k_start+1}
        C is 2x2: indices (k, i) where k in {k_start, k_start+1}, i in {0,1}
        
        Global indexing:
        A[i*3 + j]
        B[j*7 + k]
        C[k*2 + i]
        """
        # A indices
        a00 = 0 * 3 + 0  # 0
        a01 = 0 * 3 + 1  # 1
        a10 = 1 * 3 + 0  # 3
        a11 = 1 * 3 + 1  # 4
        
        # B indices
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        # C indices
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1 = (A00 + A11)(B00 + B11) -> contributes to C00, C11
        add_term([a00, a11], [b00, b11], [c00, c11])
        
        # M2 = (A10 + A11)(B00) -> contributes to C10, -C11
        add_term([a10, a11], [b00], [c10, c11], signs_w=[1, -1])
        
        # M3 = (A00)(B01 - B11) -> contributes to C01, C11
        add_term([a00], [b01, b11], [c01, c11], signs_v=[1, -1])
        
        # M4 = (A11)(B10 - B00) -> contributes to C00, C10
        add_term([a11], [b10, b00], [c00, c10], signs_v=[1, -1])
        
        # M5 = (A00 + A01)(B11) -> contributes to -C00, C01
        add_term([a00, a01], [b11], [c00, c01], signs_w=[-1, 1])
        
        # M6 = (A10 - A00)(B00 + B01) -> contributes to C11
        add_term([a10, a00], [b00, b01], [c11], signs_u=[1, -1])
        
        # M7 = (A01 - A11)(B10 + B11) -> contributes to C00
        add_term([a01, a11], [b10, b11], [c00], signs_u=[1, -1])
    
    # Add 3 Strassen blocks for k=0,1; k=2,3; k=4,5
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Schoolbook block for k=6 (2x2x1)
    # C06 = A00*B06 + A01*B16
    # C16 = A10*B06 + A11*B16
    b06 = 0 * 7 + 6  # 6
    b16 = 1 * 7 + 6  # 13
    c06 = 6 * 2 + 0  # 12
    c16 = 6 * 2 + 1  # 13
    
    # C06 terms
    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    
    # C16 terms
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])
    
    # Part 2: j=2 (2x1x7 schoolbook)
    # C0k = A02*B2k
    # C1k = A12*B2k
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        # C0k = A02 * B2k
        add_term([2], [b2k], [c0k])
        
        # C1k = A12 * B2k
        add_term([5], [b2k], [c1k])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
