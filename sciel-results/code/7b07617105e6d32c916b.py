"""Bootstrap spawn: Direct implementation of rank 189 decomposition for 6x6 matrix multiplication using 2x2 Strassen blocks."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 189: 9 blocks of 2x2 Strassen (rank 7 each)
    rank = 189
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen terms for 2x2 matrix multiplication
    # Each term has U (A-side), V (B-side), W (C-side) coefficients
    # Local indices: (row, col) in {0,1}^2
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        {'U': [(0, 0, 1), (1, 1, 1)], 
         'V': [(0, 0, 1), (1, 1, 1)], 
         'W': [(0, 0, 1), (1, 1, 1)]},
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        {'U': [(1, 0, 1), (1, 1, 1)], 
         'V': [(0, 0, 1)], 
         'W': [(1, 0, 1), (1, 1, -1)]},
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        {'U': [(0, 0, 1)], 
         'V': [(0, 1, 1), (1, 1, -1)], 
         'W': [(0, 1, 1), (1, 1, 1)]},
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        {'U': [(1, 1, 1)], 
         'V': [(1, 0, 1), (0, 0, -1)], 
         'W': [(0, 0, 1), (1, 0, 1)]},
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        {'U': [(0, 0, 1), (0, 1, 1)], 
         'V': [(1, 1, 1)], 
         'W': [(0, 0, -1), (0, 1, 1)]},
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        {'U': [(1, 0, 1), (0, 0, -1)], 
         'V': [(0, 0, 1), (0, 1, 1)], 
         'W': [(1, 1, 1)]},
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        {'U': [(0, 1, 1), (1, 1, -1)], 
         'V': [(1, 0, 1), (1, 1, 1)], 
         'W': [(0, 0, 1)]}
    ]
    
    # 6x6 = 3x3 grid of 2x2 blocks
    for i in range(3):  # A row block
        for k in range(3):  # A col block / B row block
            for j in range(3):  # B col block / C col block
                for t in range(7):
                    col_idx = (i * 9 + k * 3 + j) * 7 + t
                    term = strassen_terms[t]
                    
                    # A-side: block at (i*2, k*2), local offset (lr, lc)
                    for lr, lc, coeff in term['U']:
                        u_row = (i * 2 + lr) * m + (k * 2 + lc)
                        U[u_row, col_idx] += coeff
                    
                    # B-side: block at (k*2, j*2), local offset (lr, lc)
                    for lr, lc, coeff in term['V']:
                        v_row = (k * 2 + lr) * p + (j * 2 + lc)
                        V[v_row, col_idx] += coeff
                    
                    # C-side: block at (i*2, j*2), local offset (lr, lc)
                    # W index convention: W[k*n + i, r] where k is col, i is row
                    for lr, lc, coeff in term['W']:
                        w_row = (j * 2 + lc) * n + (i * 2 + lr)
                        W[w_row, col_idx] += coeff
    
    # Convert to lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
