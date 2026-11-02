import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def construct():
    """
    Returns a rank-21 decomposition for M_{2,2,6}.
    This uses Strassen's algorithm on three 2x2 blocks.
    """
    n, m, p = DIMENSIONS  # 2, 2, 6
    
    # Strassen formula coefficients for 2x2 matrix multiplication
    # Indices for A (2x2): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Indices for B (2x2): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Indices for C (2x2): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # The 7 Strassen products
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B10 - B00)
    # M3 = A11(B01 - B11)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A11 - A10)(B10 + B11)
    
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 - M5
    
    # U coeffs for A (indices 0,1,2,3)
    # V coeffs for B (indices 0,1,2,3)
    # W coeffs for C (indices (row, col) -> global index)
    
    strassen_terms = [
        # r=0: M0
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # r=1: M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # r=2: M2
        ({0: 1}, {2: 1, 0: -1}, {1: 1, 3: 1}),
        # r=3: M3
        ({3: 1}, {1: 1, 3: -1}, {1: 1, 2: 1}),
        # r=4: M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # r=5: M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # r=6: M6
        ({3: 1, 2: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    rank = 21
    # U: (n*m, rank) = (4, 21)
    # V: (m*p, rank) = (12, 21)
    # W: (p*n, rank) = (12, 21)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r_idx = 0
    # We split the 2x6 multiplication into three 2x2 multiplications
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            
            # Fill U (same for all blocks, indices in A are 0..3)
            for a_idx, val in u_coeffs.items():
                U[a_idx][r_idx] = val
                
            # Fill V
            # B indices: local 0..3 for the 2x2 block at columns 2b, 2b+1
            # Global B index: row * p + col
            # Local B indices:
            # 0: (0,0) -> row 0, col 0 (local) -> global row 0, col 2b
            # 1: (0,1) -> row 0, col 1 (local) -> global row 0, col 2b+1
            # 2: (1,0) -> row 1, col 0 (local) -> global row 1, col 2b
            # 3: (1,1) -> row 1, col 1 (local) -> global row 1, col 2b+1
            for l_idx, val in v_coeffs.items():
                row = l_idx // 2
                col_local = l_idx % 2
                col_global = 2 * b + col_local
                v_idx = row * p + col_global
                V[v_idx][r_idx] = val
                
            # Fill W
            # C indices: local (row, col) for the 2x2 block at columns 2b, 2b+1
            # Global W index: k*n + i, where k is global col, i is row
            # Local C indices in dict keys are flattened:
            # 0: (0,0)
            # 1: (0,1)
            # 2: (1,0)
            # 3: (1,1)
            for c_idx, val in w_coeffs.items():
                row_local = c_idx // 2
                col_local = c_idx % 2
                col_global = 2 * b + col_local
                # W index: k * n + i = col_global * n + row_local
                w_idx = col_global * n + row_local
                W[w_idx][r_idx] = val
                
            r_idx += 1
            
    return {"U": U, "V": V, "W": W}
