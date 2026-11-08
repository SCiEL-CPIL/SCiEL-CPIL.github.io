import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank calculation:
    # The multiplication is C = A * B where A is 2x3, B is 3x7, C is 2x7.
    # We split B into column pairs and a single column.
    # Columns 0-1, 2-3, 4-5 are pairs. Column 6 is single.
    # For each pair of columns (2 columns), we compute A * B_pair (2x3 * 3x2 -> 2x2).
    # This is a 2x3x2 tensor.
    # Rank of 2x3x2:
    #   Split A into A_2x2 and A_2x1.
    #   A_2x2 * B_2x2 (2x2x2) has rank 7 (Strassen).
    #   A_2x1 * B_1x2 (2x1x2) has rank 4 (Schoolbook).
    #   Total rank per pair = 11.
    # There are 3 pairs, so 3 * 11 = 33.
    # For the single column 6, we compute A * B_single (2x3 * 3x1 -> 2x1).
    # This is a 2x3x1 tensor.
    # Rank is 2*3 = 6 (Schoolbook).
    # Total rank = 33 + 6 = 39.
    
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]   # 6 x 39
    V = [[0] * rank for _ in range(m * p)]   # 21 x 39
    W = [[0] * rank for _ in range(p * n)]   # 14 x 39
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, u_signs=None, v_signs=None, w_signs=None):
        """Add a rank-1 term with optional signs."""
        nonlocal col
        if u_signs is None:
            u_signs = [1] * len(u_indices)
        if v_signs is None:
            v_signs = [1] * len(v_indices)
        if w_signs is None:
            w_signs = [1] * len(w_indices)
        
        for idx, s in zip(u_indices, u_signs):
            U[idx][col] += s
        for idx, s in zip(v_indices, v_signs):
            V[idx][col] += s
        for idx, s in zip(w_indices, w_signs):
            W[idx][col] += s
        col += 1

    def add_strassen_block(k_start):
        """
        Add Strassen decomposition for 2x2 * 2x2 block.
        Handles A[:, 0:2] * B[0:2, k_start:k_start+2].
        """
        # A indices (2x2 part)
        a00, a01, a10, a11 = 0, 1, 3, 4
        
        # B indices (2x2 part)
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        # C indices (2x2 part)
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        add_term([a00, a11], [b00, b11], [c00, c11])
        
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        add_term([a10, a11], [b00], [c10, c11], w_signs=[1, -1])
        
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        add_term([a00], [b01, b11], [c01, c11], v_signs=[1, -1])
        
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        add_term([a11], [b10, b00], [c00, c10], v_signs=[1, -1])
        
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        add_term([a00, a01], [b11], [c00, c01], w_signs=[-1, 1])
        
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        add_term([a10, a00], [b00, b01], [c11], u_signs=[1, -1])
        
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        add_term([a01, a11], [b10, b11], [c00], u_signs=[1, -1])

    def add_schoolbook_pair(k_start):
        """
        Add schoolbook terms for A[:, 2] * B[2, k_start:k_start+2].
        This is a 2x1 * 1x2 multiplication.
        C0k += A02 * B2k
        C1k += A12 * B2k
        """
        a02, a12 = 2, 5
        
        b20 = 2 * 7 + k_start
        b21 = 2 * 7 + k_start + 1
        
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # A02 * B20 -> C00
        add_term([a02], [b20], [c00])
        # A02 * B21 -> C01
        add_term([a02], [b21], [c01])
        # A12 * B20 -> C10
        add_term([a12], [b20], [c10])
        # A12 * B21 -> C11
        add_term([a12], [b21], [c11])

    def add_schoolbook_single(k):
        """
        Add schoolbook terms for A[:, :] * B[:, k].
        This is a 2x3 * 3x1 multiplication.
        C0k += A00 B0k + A01 B1k + A02 B2k
        C1k += A10 B0k + A11 B1k + A12 B2k
        """
        # A indices
        a00, a01, a02 = 0, 1, 2
        a10, a11, a12 = 3, 4, 5
        
        # B indices
        b0k = 0 * 7 + k
        b1k = 1 * 7 + k
        b2k = 2 * 7 + k
        
        # C indices
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        add_term([a00], [b0k], [c0k])
        add_term([a01], [b1k], [c0k])
        add_term([a02], [b2k], [c0k])
        
        add_term([a10], [b0k], [c1k])
        add_term([a11], [b1k], [c1k])
        add_term([a12], [b2k], [c1k])

    # Process column pairs 0-1, 2-3, 4-5
    for k_start in [0, 2, 4]:
        add_strassen_block(k_start)
        add_schoolbook_pair(k_start)
        
    # Process single column 6
    add_schoolbook_single(6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
