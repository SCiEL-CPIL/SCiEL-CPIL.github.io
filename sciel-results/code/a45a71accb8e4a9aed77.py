# EVOLVE-BLOCK-START
import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def construct():
    """
    Returns a rank-21 decomposition for M_{2,2,6} using Strassen's algorithm
    applied to three 2x2 column blocks of B.
    
    The decomposition is:
    U: shape (n*m, rank) = (4, 21)
    V: shape (m*p, rank) = (12, 21)
    W: shape (p*n, rank) = (12, 21)
    
    Indexing convention for W: W[k*n + i, r] where k is column index in C,
    i is row index in C, r is rank index.
    """
    n, m, p = DIMENSIONS  # 2, 2, 6
    rank = 21
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7-term algorithm for 2x2 matrix multiplication
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A @ B
    
    # U indices: A00=0, A01=1, A10=2, A11=3
    # V local indices within a block: B00=0, B01=1, B10=2, B11=3
    # W local indices within a block: C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)
    
    # Strassen multipliers:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B00)
    # M4 = (A11)(B01-B11)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A11-A10)(B10+B11)
    
    # Result:
    # C00 = M1 - M5 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M4
    
    # Each term is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping A index -> coefficient
    # v_coeffs: dict mapping local B index -> coefficient  
    # w_coeffs: dict mapping (row, col) local C index -> coefficient
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        # M2 = (A10+A11)(B00)
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        # M3 = (A00)(B10-B00)
        ({0: 1}, {2: 1, 0: -1}, {(0, 1): 1, (1, 1): 1}),
        # M4 = (A11)(B01-B11)
        ({3: 1}, {1: 1, 3: -1}, {(0, 0): 1, (1, 0): 1}),
        # M5 = (A00+A01)(B11)
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        # M6 = (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        # M7 = (A11-A10)(B10+B11)
        ({3: 1, 2: -1}, {2: 1, 3: 1}, {(0, 0): -1}),
    ]
    
    # Apply to each of the 3 column blocks
    for b in range(3):
        for t in range(7):
            r = b * 7 + t
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            
            # Fill U column r
            for idx, val in u_coeffs.items():
                U[idx, r] = val
            
            # Fill V column r
            # Local B index l maps to global V index:
            # l=0 (B00) -> row 0, col 2b -> index 0*6 + 2b = 2b
            # l=1 (B01) -> row 0, col 2b+1 -> index 0*6 + 2b+1 = 2b+1
            # l=2 (B10) -> row 1, col 2b -> index 1*6 + 2b = 6+2b
            # l=3 (B11) -> row 1, col 2b+1 -> index 1*6 + 2b+1 = 6+2b+1
            for l, val in v_coeffs.items():
                if l == 0:
                    v_idx = 2 * b
                elif l == 1:
                    v_idx = 2 * b + 1
                elif l == 2:
                    v_idx = 6 + 2 * b
                elif l == 3:
                    v_idx = 6 + 2 * b + 1
                V[v_idx, r] = val
            
            # Fill W column r
            # W index = k*n + i where k is column in C, i is row in C
            # Local C (i, k_local) maps to global:
            # k_global = 2*b + k_local
            # W index = k_global * n + i = (2*b + k_local) * 2 + i
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx, r] = val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
