"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for 6x6 matrix multiplication.
    
    Strategy: Use Strassen's algorithm at the block level (2x2 blocks of 3x3 matrices),
    where each of the 7 Strassen products is computed via the naive 3x3x3 decomposition (rank 27).
    Total rank = 7 * 27 = 189.
    
    The tensor T for matrix multiplication C = A * B has entries T[i, j, k] such that
    C[i, k] = sum_j A[i, j] B[j, k].
    
    The decomposition factors U (n*m x rank), V (m*p x rank), W (p*n x rank) satisfy:
    T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    where a = i*m + j (index into A), b = j*p + k (index into B), c = k*n + i (index into C).
    Note: The task says W follows the trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # We partition into 2x2 blocks of 3x3 submatrices.
    # Block indices: (bi, bj) where bi, bj in {0, 1}
    # Within each block, indices are 0..2.
    
    # Global indices:
    # A[i, j] where i = bi*3 + ii, j = bj*3 + jj
    # B[j, k] where j = bj*3 + jj, k = bk*3 + kk
    # C[i, k] where i = bi*3 + ii, k = bk*3 + kk
    
    # Strassen's algorithm for 2x2 block matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each "block product" like (A_block) * (B_block) is a 3x3 * 3x3 -> 3x3 product.
    # For each such product, we use the naive decomposition of rank 27.
    # The naive decomposition for 3x3: for each (ii, jj, kk) in 0..2^3, we have a term
    # A[ii, jj] * B[jj, kk] * C[ii, kk].
    
    # So for each of the 7 Strassen terms, we have 27 scalar terms, total 189 terms.
    
    # Let's build U, V, W.
    # U has shape (n*m, rank) = (36, 189)
    # V has shape (m*p, rank) = (36, 189)
    # W has shape (p*n, rank) = (36, 189)
    
    rank = 189
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Define the Strassen terms. Each term is a product of a linear combination of A blocks
    # and a linear combination of B blocks, contributing to specific C blocks with signs.
    
    # We'll define each of the 7 terms as:
    # term_id: 0..6
    # For each term, we need:
    #   - The A-side: a linear combination of A blocks. Each A block is identified by (bi, bj).
    #     The coefficient for each (bi, bj) pair.
    #   - The B-side: a linear combination of B blocks. Each B block is identified by (bj, bk).
    #     The coefficient for each (bj, bk) pair.
    #   - The C-side: contributions to C blocks. Each C block is identified by (bi, bk).
    #     The coefficient for each (bi, bk) pair.
    
    # Strassen terms:
    # M1 = (A00 + A11) * (B00 + B11)
    #   A-side: A00 coeff +1, A11 coeff +1
    #   B-side: B00 coeff +1, B11 coeff +1
    #   C-side: C00 coeff +1, C11 coeff +1
    #
    # M2 = (A10 + A11) * B00
    #   A-side: A10 coeff +1, A11 coeff +1
    #   B-side: B00 coeff +1
    #   C-side: C10 coeff +1, C11 coeff -1
    #
    # M3 = A00 * (B01 - B11)
    #   A-side: A00 coeff +1
    #   B-side: B01 coeff +1, B11 coeff -1
    #   C-side: C01 coeff +1, C11 coeff +1
    #
    # M4 = A11 * (B10 - B00)
    #   A-side: A11 coeff +1
    #   B-side: B10 coeff +1, B00 coeff -1
    #   C-side: C00 coeff +1, C10 coeff +1
    #
    # M5 = (A00 + A01) * B11
    #   A-side: A00 coeff +1, A01 coeff +1
    #   B-side: B11 coeff +1
    #   C-side: C00 coeff -1, C01 coeff +1
    #
    # M6 = (A10 - A00) * (B00 + B01)
    #   A-side: A10 coeff +1, A00 coeff -1
    #   B-side: B00 coeff +1, B01 coeff +1
    #   C-side: C11 coeff +1
    #
    # M7 = (A01 - A11) * (B10 + B11)
    #   A-side: A01 coeff +1, A11 coeff -1
    #   B-side: B10 coeff +1, B11 coeff +1
    #   C-side: C00 coeff +1, C11 coeff +1
    
    # Let's encode these.
    # A blocks: indexed by (bi, bj) in {0,1}x{0,1}
    # B blocks: indexed by (bj, bk) in {0,1}x{0,1}
    # C blocks: indexed by (bi, bk) in {0,1}x{0,1}
    
    strassen_terms = [
        # (A_coefs: dict from (bi,bj) to coef, B_coefs: dict from (bj,bk) to coef, C_coefs: dict from (bi,bk) to coef)
        {
            'A': {(0,0): 1, (1,1): 1},
            'B': {(0,0): 1, (1,1): 1},
            'C': {(0,0): 1, (1,1): 1}
        },
        {
            'A': {(1,0): 1, (1,1): 1},
            'B': {(0,0): 1},
            'C': {(1,0): 1, (1,1): -1}
        },
        {
            'A': {(0,0): 1},
            'B': {(0,1): 1, (1,1): -1},
            'C': {(0,1): 1, (1,1): 1}
        },
        {
            'A': {(1,1): 1},
            'B': {(1,0): 1, (0,0): -1},
            'C': {(0,0): 1, (1,0): 1}
        },
        {
            'A': {(0,0): 1, (0,1): 1},
            'B': {(1,1): 1},
            'C': {(0,0): -1, (0,1): 1}
        },
        {
            'A': {(1,0): 1, (0,0): -1},
            'B': {(0,0): 1, (0,1): 1},
            'C': {(1,1): 1}
        },
        {
            'A': {(0,1): 1, (1,1): -1},
            'B': {(1,0): 1, (1,1): 1},
            'C': {(0,0): 1, (1,1): 1}
        }
    ]
    
    column = 0
    for term in strassen_terms:
        A_coefs = term['A']
        B_coefs = term['B']
        C_coefs = term['C']
        
        # For each (ii, jj, kk) in 0..2^3, we have a scalar term for this Strassen product.
        for ii in range(3):
            for jj in range(3):
                for kk in range(3):
                    # This scalar term corresponds to:
                    # A-side: sum over (bi, bj) in A_coefs of coef_A * A[bi*3+ii, bj*3+jj]
                    # B-side: sum over (bj, bk) in B_coefs of coef_B * B[bj*3+jj, bk*3+kk]
                    # C-side: sum over (bi, bk) in C_coefs of coef_C * C[bi*3+ii, bk*3+kk]
                    
                    # For the U factor: U[a, column] where a = i*m + j = (bi*3+ii)*6 + (bj*3+jj)
                    # The coefficient is coef_A for each (bi, bj).
                    for (bi, bj), coef_a in A_coefs.items():
                        i = bi * 3 + ii
                        j = bj * 3 + jj
                        a_idx = i * m + j
                        U[a_idx, column] += coef_a
                    
                    # For the V factor: V[b, column] where b = j*p + k = (bj*3+jj)*6 + (bk*3+kk)
                    # The coefficient is coef_B for each (bj, bk).
                    for (bj, bk), coef_b in B_coefs.items():
                        j = bj * 3 + jj
                        k = bk * 3 + kk
                        b_idx = j * p + k
                        V[b_idx, column] += coef_b
                    
                    # For the W factor: W[c, column] where c = k*n + i = (bk*3+kk)*6 + (bi*3+ii)
                    # The coefficient is coef_C for each (bi, bk).
                    for (bi, bk), coef_c in C_coefs.items():
                        k = bk * 3 + kk
                        i = bi * 3 + ii
                        c_idx = k * n + i
                        W[c_idx, column] += coef_c
                    
                    column += 1
    
    # Convert to lists as expected by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
