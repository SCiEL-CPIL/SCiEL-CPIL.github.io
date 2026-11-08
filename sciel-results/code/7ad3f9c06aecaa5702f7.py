import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 3 Strassen blocks (2x2x2) for k=0,1; 2,3; 4,5 -> 3 * 7 = 21
    # 1 Schoolbook block (2x2x1) for k=6, j=0,1 -> 4
    # j=2 part: 3 pairs (k=0,1), (2,3), (4,5) each use 3 rank-1 terms = 9
    # j=2, k=6: 2 rank-1 terms
    # Total: 21 + 4 + 9 + 2 = 36
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank overflow")
        for idx, coeff in zip(u_indices, u_coeffs):
            U[idx][col] += coeff
        for idx, coeff in zip(v_indices, v_coeffs):
            V[idx][col] += coeff
        for idx, coeff in zip(w_indices, w_coeffs):
            W[idx][col] += coeff
        col += 1

    def add_strassen_block(k_start):
        # A indices: A00=0, A01=1, A10=3, A11=4
        a00, a01, a10, a11 = 0, 1, 3, 4
        
        # B indices
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        # C indices: W[k*n+i, r] -> index = k*2 + i
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1])
        
        # M2: (A10+A11)B00 -> C10, -C11
        add_term([a10, a11], [1, 1], [b00], [1], [c10, c11], [1, -1])
        
        # M3: A00(B01-B11) -> C01, C11
        add_term([a00], [1], [b01, b11], [1, -1], [c01, c11], [1, 1])
        
        # M4: A11(B10-B00) -> C00, C10
        add_term([a11], [1], [b10, b00], [1, -1], [c00, c10], [1, 1])
        
        # M5: (A00+A01)B11 -> -C00, C01
        add_term([a00, a01], [1, 1], [b11], [1], [c00, c01], [-1, 1])
        
        # M6: (A10-A00)(B00+B01) -> C11
        add_term([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1])
        
        # M7: (A01-A11)(B10+B11) -> C00
        add_term([a01, a11], [1, -1], [b10, b11], [1, 1], [c00], [1])

    # Add 3 Strassen blocks for k=0, 2, 4
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Block for k=6, j=0,1: Schoolbook 2x2x1
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [1], [b06], [1], [c06], [1])
    add_term([1], [1], [b16], [1], [c06], [1])
    add_term([3], [1], [b06], [1], [c16], [1])
    add_term([4], [1], [b16], [1], [c16], [1])
    
    # j=2 part: A[0:2, 2] * B[2, 0:7] -> C[0:2, 0:7]
    # A02 = index 2, A12 = index 5
    # B[2, k] = index 14 + k
    # C[i, k] = index k*2 + i
    
    # For each pair (k, k+1), use 3 rank-1 terms:
    # Term 1: A02 * B2k -> C0k
    # Term 2: A02 * B2(k+1) -> C0(k+1)
    # Term 3: A12 * (B2k + B2(k+1)) -> C1k + C1(k+1)
    for k_pair in range(0, 6, 2):
        k0 = k_pair
        k1 = k_pair + 1
        b2k0 = 2 * 7 + k0
        b2k1 = 2 * 7 + k1
        c0k0 = k0 * 2 + 0
        c0k1 = k1 * 2 + 0
        c1k0 = k0 * 2 + 1
        c1k1 = k1 * 2 + 1
        
        # Term 1: A02 * B2k0 -> C0k0
        add_term([2], [1], [b2k0], [1], [c0k0], [1])
        
        # Term 2: A02 * B2k1 -> C0k1
        add_term([2], [1], [b2k1], [1], [c0k1], [1])
        
        # Term 3: A12 * (B2k0 + B2k1) -> C1k0 + C1k1
        add_term([5], [1], [b2k0, b2k1], [1, 1], [c1k0, c1k1], [1, 1])
    
    # j=2, k=6: 2 terms
    b26 = 2 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([2], [1], [b26], [1], [c06], [1])
    add_term([5], [1], [b26], [1], [c16], [1])
    
    if col != rank:
        # Pad with zeros if needed
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
