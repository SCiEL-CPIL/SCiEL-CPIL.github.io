"""
Program implementing Strassen-based rank minimization for 4x4x6 matrix multiplication.

Uses the optimal 49-term Strassen decomposition for the 4x4x4 part and
a 28-term decomposition for the 4x4x2 part, for a total rank of 77.
"""
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 terms
    strassen_2x2_terms = [
        {'A': [[1, 0], [0, 1]], 'B': [[1, 0], [0, 1]], 'C': [[1, 0], [0, 1]]},
        {'A': [[0, 0], [1, 1]], 'B': [[1, 0], [0, 0]], 'C': [[0, 0], [1, -1]]},
        {'A': [[1, 0], [0, 0]], 'B': [[0, 1], [0, -1]], 'C': [[0, 1], [0, 1]]},
        {'A': [[0, 0], [0, 1]], 'B': [[-1, 0], [1, 0]], 'C': [[1, 0], [1, 0]]},
        {'A': [[1, 1], [0, 0]], 'B': [[0, 0], [0, 1]], 'C': [[-1, 1], [0, 0]]},
        {'A': [[-1, 0], [1, 0]], 'B': [[1, 1], [0, 0]], 'C': [[0, 0], [0, 1]]},
        {'A': [[0, 1], [0, -1]], 'B': [[0, 0], [1, 1]], 'C': [[1, 0], [0, 0]]},
    ]
    
    terms = []
    
    def add_product_of_block_sums(a_blocks, b_blocks, c_blocks):
        for t in strassen_2x2_terms:
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            for i in range(2):
                for j in range(2):
                    for (r, c, coeff) in a_blocks:
                        u[(r + i) * m + (c + j)] += coeff * t['A'][i][j]
                    for (r, c, coeff) in b_blocks:
                        v[(r + i) * p + (c + j)] += coeff * t['B'][i][j]
                    for (r, c, cc) in c_blocks:
                        w[(c + j) * n + (r + i)] += cc * t['C'][i][j]
            
            terms.append((u, v, w))
    
    # 4x4x4 part
    add_product_of_block_sums(
        [(0, 0, 1), (2, 2, 1)],
        [(0, 0, 1), (2, 2, 1)],
        [(0, 0, 1), (2, 2, 1)]
    )
    add_product_of_block_sums(
        [(2, 0, 1), (2, 2, 1)],
        [(0, 0, 1)],
        [(2, 0, 1), (2, 2, -1)]
    )
    add_product_of_block_sums(
        [(0, 0, 1)],
        [(0, 2, 1), (2, 2, -1)],
        [(0, 2, 1), (2, 2, 1)]
    )
    add_product_of_block_sums(
        [(2, 2, 1)],
        [(2, 0, 1), (0, 0, -1)],
        [(0, 0, 1), (2, 0, 1)]
    )
    add_product_of_block_sums(
        [(0, 0, 1), (0, 2, 1)],
        [(2, 2, 1)],
        [(0, 0, -1), (0, 2, 1)]
    )
    add_product_of_block_sums(
        [(2, 0, 1), (0, 0, -1)],
        [(0, 0, 1), (0, 2, 1)],
        [(2, 2, 1)]
    )
    add_product_of_block_sums(
        [(0, 2, 1), (2, 2, -1)],
        [(2, 0, 1), (2, 2, 1)],
        [(0, 0, 1)]
    )
    
    # 4x4x2 part
    # A00*B_top
    add_product_of_block_sums(
        [(0, 0, 1)],
        [(0, 4, 1)],
        [(0, 4, 1)]
    )
    # A01*B_bot
    add_product_of_block_sums(
        [(0, 2, 1)],
        [(2, 4, 1)],
        [(0, 4, 1)]
    )
    # A10*B_top
    add_product_of_block_sums(
        [(2, 0, 1)],
        [(0, 4, 1)],
        [(2, 4, 1)]
    )
    # A11*B_bot
    add_product_of_block_sums(
        [(2, 2, 1)],
        [(2, 4, 1)],
        [(2, 4, 1)]
    )
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k, (u, v, w) in enumerate(terms):
        for i in range(n * m):
            U[i][k] = u[i]
        for i in range(m * p):
            V[i][k] = v[i]
        for i in range(p * n):
            W[i][k] = w[i]
    
    return {"U": U, "V": V, "W": W}
