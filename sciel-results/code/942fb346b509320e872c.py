"""Direct implementation of rank 189 decomposition for 6x6 matrix multiplication."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = start_time + 120.0
    SAFETY_MARGIN = 10.0

    # Rank-189 solution: 27 blocks of 2x2 Strassen (7 terms each)
    # 6x6 MM split into 3x3 blocks of 2x2 submatrices
    # Each 2x2 block multiplication uses Strassen (rank 7)
    # Total: 27 * 7 = 189
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 MM
    # C = AB, C_{ik} = sum_j A_{ij} B_{jk}
    # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
    # M2 = (A10 + A11)B00       -> C10 += M2, C11 -= M2
    # M3 = A00(B01 - B11)       -> C01 += M3, C11 += M3
    # M4 = A11(B10 - B00)       -> C00 += M4, C10 += M4
    # M5 = (A00 + A01)B11       -> C00 -= M5, C01 += M5
    # M6 = (A10 - A00)(B00 + B01) -> C11 += M6
    # M7 = (A01 - A11)(B10 + B11) -> C00 += M7
    
    strassen_terms = [
        {
            'U': [(0, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]
        },
        {
            'U': [(1, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(1, 0, 1), (1, 1, -1)]
        },
        {
            'U': [(0, 0, 1)],
            'V': [(0, 1, 1), (1, 1, -1)],
            'W': [(0, 1, 1), (1, 1, 1)]
        },
        {
            'U': [(1, 1, 1)],
            'V': [(1, 0, 1), (0, 0, -1)],
            'W': [(0, 0, 1), (1, 0, 1)]
        },
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1)],
            'W': [(0, 0, -1), (0, 1, 1)]
        },
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 0, 1), (0, 1, 1)],
            'W': [(1, 1, 1)]
        },
        {
            'U': [(0, 1, 1), (1, 1, -1)],
            'V': [(1, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1)]
        }
    ]
    
    # Iterate over 3x3 blocks of 2x2 submatrices
    # Block (i,k) in C = sum_j A(i,j) B(j,k)
    # i,j,k in {0,1,2}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for t in range(7):
                    col_idx = (i * 9 + j * 3 + k) * 7 + t
                    term = strassen_terms[t]
                    # A block at (i,j): rows 2i..2i+1, cols 2j..2j+1
                    # B block at (j,k): rows 2j..2j+1, cols 2k..2k+1
                    # C block at (i,k): rows 2i..2i+1, cols 2k..2k+1
                    for lr, lc, coeff in term['U']:
                        u_row = (2 * i + lr) * m + (2 * j + lc)
                        U[u_row][col_idx] += coeff
                    for lr, lc, coeff in term['V']:
                        v_row = (2 * j + lr) * p + (2 * k + lc)
                        V[v_row][col_idx] += coeff
                    for lr, lc, coeff in term['W']:
                        # W index: col * n + row (trace convention)
                        w_row = (2 * k + lc) * n + (2 * i + lr)
                        W[w_row][col_idx] += coeff
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
