import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition:
    # 7 terms, each is a 3x3 matrix multiplication.
    # 3x3 schoolbook rank is 27.
    # Total rank = 7 * 27 = 189.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block offsets for 3x3 blocks in 6x6 matrices
    # Block indices: 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # Mapping to (row_offset, col_offset)
    block_offsets = {
        0: (0, 0), # 11
        1: (0, 3), # 12
        2: (3, 0), # 21
        3: (3, 3)  # 22
    }
    
    # Strassen terms
    # Each term is (alpha_dict, beta_dict, gamma_dict)
    # alpha: coefficients for A blocks
    # beta: coefficients for B blocks
    # gamma: coefficients for C blocks (contribution to output)
    
    # M1 = (A11 + A22)(B11 + B22)
    # C11 += M1, C22 += M1
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A21 + A22)B11
        # C21 += M2, C22 -= M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A11(B12 - B22)
        # C12 += M3, C22 += M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A22(B21 - B11)
        # C11 += M4, C21 += M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A11 + A12)B22
        # C11 -= M5, C12 += M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A21 - A11)(B11 + B12)
        # C22 += M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A12 - A22)(B21 + B22)
        # C11 += M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    col = 0
    for alpha, beta, gamma in terms:
        # Iterate over 3x3 schoolbook terms
        # r: summation index (0..2)
        # i: row index of X (0..2)
        # j: col index of Y (0..2)
        for r in range(3):
            for i in range(3):
                for j in range(3):
                    # Update U
                    # U picks up X_{i, r}
                    # X_{i, r} = sum_{bA} alpha[bA] * A_{bA, i, r}
                    for bA, coeff_a in alpha.items():
                        rA, cA = block_offsets[bA]
                        u_row = rA + i
                        u_col = cA + r
                        u_idx = u_row * m + u_col
                        U[u_idx][col] += coeff_a
                    
                    # Update V
                    # V picks up Y_{r, j}
                    # Y_{r, j} = sum_{bB} beta[bB] * B_{bB, r, j}
                    for bB, coeff_b in beta.items():
                        rB, cB = block_offsets[bB]
                        v_row = rB + r
                        v_col = cB + j
                        v_idx = v_row * p + v_col
                        V[v_idx][col] += coeff_b
                    
                    # Update W
                    # W accumulates Z_{i, j} into C
                    # Z_{i, j} contributes to C_{bC, i, j} with coeff gamma[bC]
                    for bC, coeff_c in gamma.items():
                        rC, cC = block_offsets[bC]
                        w_row_global = rC + i
                        w_col_global = cC + j
                        # W index is col * n + row
                        w_idx = w_col_global * n + w_row_global
                        W[w_idx][col] += coeff_c
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
