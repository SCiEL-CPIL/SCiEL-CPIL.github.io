import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    # The current best known rank for 2x4 * 4x4 is 28 (4 * 7 Strassen blocks).
    # We will verify this construction.
    
    # Indexing conventions:
    # U: shape (n*m, rank) = (8, rank). Index i*m + j corresponds to A[i, j].
    # V: shape (m*p, rank) = (16, rank). Index j*p + k corresponds to B[j, k].
    # W: shape (p*n, rank) = (8, rank). Index k*n + i corresponds to C[i, k] (Trace convention W[k*n+i]).
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen's 7 products for 2x2 matrix multiplication C = A * B
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    
    # C00 = M1 - M4 + M5 - M6 + M7
    # C01 = M3 + M5 - M7
    # C10 = M2 + M4 - M5 - M6
    # C11 = M1 + M3 - M2 + M6
    
    # We define the 7 patterns as lists of (index, coeff) for U, V, W
    # Local indices for 2x2 block:
    # A: 0=a00, 1=a01, 2=a10, 3=a11
    # B: 0=b00, 1=b01, 2=b10, 3=b11
    # C: 0=c00, 1=c01, 2=c10, 3=c11
    
    strassen_patterns = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, -1), (2, 1)]), # C00 gets -M4, C10 gets +M4
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1), (2, -1)]), # C00 gets -M5, C01 gets +M5, C10 gets -M5
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(2, -1), (3, 1)]), # C10 gets -M6, C11 gets +M6
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (1, -1)]), # C00 gets +M7, C01 gets -M7
    ]

    # Blocks:
    # A is 2x4. Split into A1 = A[:, 0:2] and A2 = A[:, 2:4].
    # B is 4x4. Split into B11, B12, B21, B22.
    # C is 2x4. C[:, 0:2] = A1*B11 + A2*B21. C[:, 2:4] = A1*B12 + A2*B22.
    
    # Block definitions: (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    # a_col_offset: 0 or 2
    # b_row_offset: 0 or 2
    # b_col_offset: 0 or 2
    # c_col_offset: 0 or 2
    
    blocks = [
        (0, 0, 0, 0), # A1 * B11 -> C[:, 0:2]
        (2, 2, 0, 0), # A2 * B21 -> C[:, 0:2]
        (0, 0, 2, 2), # A1 * B12 -> C[:, 2:4]
        (2, 2, 2, 2), # A2 * B22 -> C[:, 2:4]
    ]
    
    col_offset = 0
    for a_off, b_row_off, b_col_off, c_off in blocks:
        for u_pat, v_pat, w_pat in strassen_patterns:
            # Fill U
            for idx, coeff in u_pat:
                # idx is local 2x2 index: 0,1,2,3
                # local row = idx // 2, local col = idx % 2
                i = idx // 2
                j = a_off + (idx % 2)
                U[i * m + j, col_offset] += coeff
            
            # Fill V
            for idx, coeff in v_pat:
                # idx is local 2x2 index
                j = b_row_off + (idx // 2)
                k = b_col_off + (idx % 2)
                V[j * p + k, col_offset] += coeff
            
            # Fill W
            for idx, coeff in w_pat:
                # idx is local 2x2 index
                i = idx // 2
                k = c_off + (idx % 2)
                # Trace convention: W[k*n + i]
                W[k * n + i, col_offset] += coeff
            
            col_offset += 1

    return {"U": U, "V": V, "W": W}
