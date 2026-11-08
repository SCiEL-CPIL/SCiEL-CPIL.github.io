import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Using block-wise Strassen decomposition
    # 6x6 matrices are partitioned into 3x3 grid of 2x2 blocks
    # Each 2x2 multiplication uses rank 7 (Strassen)
    # Total rank = 3 * 3 * 3 * 7 = 189
    rank = 189
    
    # Initialize U, V, W as lists of lists of floats/ints
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions
    # Each term is a dict with 'U', 'V', 'W' lists of (row, col, coeff)
    strassen_terms = [
        # Term 0: M1 = (A00 + A11)(B00 + B11)
        {
            'U': [(0, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]
        },
        # Term 1: M2 = (A10 + A11)B00
        {
            'U': [(1, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(1, 0, 1), (1, 1, -1)]
        },
        # Term 2: M3 = A00(B01 - B11)
        {
            'U': [(0, 0, 1)],
            'V': [(0, 1, 1), (1, 1, -1)],
            'W': [(0, 1, 1), (1, 1, 1)]
        },
        # Term 3: M4 = A11(B10 - B00)
        {
            'U': [(1, 1, 1)],
            'V': [(1, 0, 1), (0, 0, -1)],
            'W': [(0, 0, 1), (1, 0, 1)]
        },
        # Term 4: M5 = (A00 + A01)B11
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1)],
            'W': [(0, 0, -1), (0, 1, 1)]
        },
        # Term 5: M6 = (A10 - A00)(B00 + B01)
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 0, 1), (0, 1, 1)],
            'W': [(1, 1, 1)]
        },
        # Term 6: M7 = (A01 - A11)(B10 + B11)
        {
            'U': [(0, 1, 1), (1, 1, -1)],
            'V': [(1, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1)]
        }
    ]
    
    # Iterate over all block positions
    for i in range(3): # Row block of A, Row block of C
        for j in range(3): # Col block of B, Col block of C
            for k in range(3): # Col block of A, Row block of B
                for t in range(7):
                    col_idx = (i * 9 + j * 3 + k) * 7 + t
                    term = strassen_terms[t]
                    
                    # Update U
                    for lr, lc, coeff in term['U']:
                        u_row = (2 * i + lr) * m + (2 * k + lc)
                        U[u_row][col_idx] += coeff
                        
                    # Update V
                    for lr, lc, coeff in term['V']:
                        v_row = (2 * k + lr) * p + (2 * j + lc)
                        V[v_row][col_idx] += coeff
                        
                    # Update W
                    # W index convention: W[k*n + i, r] where i is row of C, k is col of C
                    for lr, lc, coeff in term['W']:
                        w_row = (2 * j + lc) * n + (2 * i + lr)
                        W[w_row][col_idx] += coeff
                        
                    # Check time
                    if time.time() > deadline:
                        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
