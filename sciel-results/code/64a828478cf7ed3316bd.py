"""Combined program for matrix multiplication 2x3x7 rank minimization."""
import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Constructive decomposition using Strassen blocks for 2x2x2 subproblems
    # and schoolbook for the remaining parts.
    # Rank: 3 * 7 (Strassen for k-pairs 0-1, 2-3, 4-5) + 4 (k=6) + 14 (j=2) = 39
    
    rank = 39
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, u_coeffs=None, v_coeffs=None, w_coeffs=None):
        nonlocal col
        if u_coeffs is None: u_coeffs = [1] * len(u_indices)
        if v_coeffs is None: v_coeffs = [1] * len(v_indices)
        if w_coeffs is None: w_coeffs = [1] * len(w_indices)
        
        for idx, c in zip(u_indices, u_coeffs):
            U[idx][col] += c
        for idx, c in zip(v_indices, v_coeffs):
            V[idx][col] += c
        for idx, c in zip(w_indices, w_coeffs):
            W[idx][col] += c
        col += 1

    def add_strassen_block(k_start):
        """
        Adds 7 rank-1 terms for multiplying a 2x2 block of A (rows 0,1)
        with a 2x2 block of B (rows 0,1, cols k_start, k_start+1)
        to produce a 2x2 block of C (rows 0,1, cols k_start, k_start+1).
        
        Indices:
        A: i in {0,1}, j in {0,1} -> idx = i*3 + j
        B: j in {0,1}, k in {k_start, k_start+1} -> idx = j*7 + k
        C: k in {k_start, k_start+1}, i in {0,1} -> idx = k*2 + i
        """
        # A indices
        a00 = 0 * 3 + 0
        a01 = 0 * 3 + 1
        a10 = 1 * 3 + 0
        a11 = 1 * 3 + 1
        
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
        
        # M1: (A00 + A11)(B00 + B11) -> C00, C11
        add_term([a00, a11], [b00, b11], [c00, c11])
        
        # M2: (A10 + A11)B00 -> C10, -C11
        add_term([a10, a11], [b00], [c10, c11], w_coeffs=[1, -1])
        
        # M3: A00(B01 - B11) -> C01, C11
        add_term([a00], [b01, b11], [c01, c11], v_coeffs=[1, -1])
        
        # M4: A11(B10 - B00) -> C00, C10
        add_term([a11], [b10, b00], [c00, c10], v_coeffs=[1, -1])
        
        # M5: (A00 + A01)B11 -> -C00, C01
        add_term([a00, a01], [b11], [c00, c01], w_coeffs=[-1, 1])
        
        # M6: (A10 - A00)(B00 + B01) -> C11
        add_term([a10, a00], [b00, b01], [c11], u_coeffs=[1, -1])
        
        # M7: (A01 - A11)(B10 + B11) -> C00
        add_term([a01, a11], [b10, b11], [c00], u_coeffs=[1, -1])

    # 1. Add 3 Strassen blocks for k-pairs (0,1), (2,3), (4,5)
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # 2. Handle k=6 for j=0,1 (2x2x1 schoolbook)
    # C06 = A00 B06 + A01 B16
    # C16 = A10 B06 + A11 B16
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0 * 3 + 0], [b06], [c06]) # A00 B06 -> C06
    add_term([0 * 3 + 1], [b16], [c06]) # A01 B16 -> C06
    add_term([1 * 3 + 0], [b06], [c16]) # A10 B06 -> C16
    add_term([1 * 3 + 1], [b16], [c16]) # A11 B16 -> C16
    
    # 3. Handle j=2 for all k (2x1x7 schoolbook)
    # Ci_k = Ai_2 B2_k
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        add_term([0 * 3 + 2], [b2k], [c0k]) # A02 B2k -> C0k
        add_term([1 * 3 + 2], [b2k], [c1k]) # A12 B2k -> C1k
        
    # Verification (optional, but good for safety)
    # The construction is exact integer arithmetic, so it should be valid.
    # We return the result immediately.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
