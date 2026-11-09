import time
import numpy as np

_DIM = (6, 6, 6)

def construct():
    n, m, p = _DIM
    # We use a 3x3 block partitioning with 2x2 Strassen blocks.
    # Block size B = 2. Number of blocks per side N_b = 3.
    B = 2
    N_b = n // B
    
    # Total rank is N_b^3 * 7
    rank = N_b * N_b * N_b * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 multiplication.
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # Result C:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each term is defined by:
    # U_coeffs: list of (local_row, local_col, coeff) for A
    # V_coeffs: list of (local_row, local_col, coeff) for B
    # W_coeffs: list of (local_row, local_col, coeff) for C
    # Local indices are 0 or 1.
    
    strassen_terms = [
        # Term 0: M1
        {
            'U': [(0, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]
        },
        # Term 1: M2
        {
            'U': [(1, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(1, 0, 1), (1, 1, -1)]
        },
        # Term 2: M3
        {
            'U': [(0, 0, 1)],
            'V': [(0, 1, 1), (1, 1, -1)],
            'W': [(0, 1, 1), (1, 1, 1)]
        },
        # Term 3: M4
        {
            'U': [(1, 1, 1)],
            'V': [(1, 0, 1), (0, 0, -1)],
            'W': [(0, 0, 1), (1, 0, 1)]
        },
        # Term 5: M5
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1)],
            'W': [(0, 0, -1), (0, 1, 1)]
        },
        # Term 5: M6
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 0, 1), (0, 1, 1)],
            'W': [(1, 1, 1)]
        },
        # Term 6: M7
        {
            'U': [(0, 1, 1), (1, 1, -1)],
            'V': [(1, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1)]
        }
    ]
    
    # Iterate over block positions
    # i: block row of A, block row of C
    # k: block col of A, block row of B
    # l: block col of B, block col of C
    for i in range(N_b):
        for k in range(N_b):
            for l in range(N_b):
                for t in range(7):
                    col_idx = (i * N_b + k) * (N_b * 7) + l * 7 + t
                    term = strassen_terms[t]
                    
                    # Update U
                    for lr, lc, coeff in term['U']:
                        # Global row in A: (i * B + lr)
                        # Global col in A: (k * B + lc)
                        # U index: row * m + col
                        u_row = (i * B + lr) * m + (k * B + lc)
                        U[u_row][col_idx] += coeff
                        
                    # Update V
                    for lr, lc, coeff in term['V']:
                        # Global row in B: (k * B + lr)
                        # Global col in B: (l * B + lc)
                        # V index: row * p + col
                        v_row = (k * B + lr) * p + (l * B + lc)
                        V[v_row][col_idx] += coeff
                        
                    # Update W
                    for lr, lc, coeff in term['W']:
                        # Global row in C: (i * B + lr)
                        # Global col in C: (l * B + lc)
                        # W index convention: W[k*n + i, r] where i is row of C, k is col of C?
                        # Prompt: W[k*n+i, r]. In standard C[i,k] = sum A[i,j]B[j,k].
                        # Here C is n x p. Rows i (0..5), Cols k (0..5).
                        # So W index is col * n + row.
                        # Row of C is i*B + lr. Col of C is l*B + lc.
                        w_row = (l * B + lc) * n + (i * B + lr)
                        W[w_row][col_idx] += coeff

    # Convert to list of lists of floats/ints as required
    # The problem accepts integer/half-integer entries.
    # Our entries are integers.
    
    return {"U": U, "V": V, "W": W}
