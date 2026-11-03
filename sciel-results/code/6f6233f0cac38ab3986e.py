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
    # Each term: sum_k U_k * V_k contributes W_k to C
    # where U_k = sum_{i,j} A_coeff[i][j] * a[i][j]
    #       V_k = sum_{i,j} B_coeff[i][j] * b[i][j]
    #       C_k = sum_{i,j} C_coeff[i][j] * c[i][j]
    strassen_2x2_terms = [
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        {'A': [[1, 0], [0, 1]], 'B': [[1, 0], [0, 1]], 'C': [[1, 0], [0, 1]]},
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        {'A': [[0, 0], [1, 1]], 'B': [[1, 0], [0, 0]], 'C': [[0, 0], [1, -1]]},
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        {'A': [[1, 0], [0, 0]], 'B': [[0, 1], [0, -1]], 'C': [[0, 1], [0, 1]]},
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        {'A': [[0, 0], [0, 1]], 'B': [[-1, 0], [1, 0]], 'C': [[1, 0], [1, 0]]},
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        {'A': [[1, 1], [0, 0]], 'B': [[0, 0], [0, 1]], 'C': [[-1, 1], [0, 0]]},
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        {'A': [[-1, 0], [1, 0]], 'B': [[1, 1], [0, 0]], 'C': [[0, 0], [0, 1]]},
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        {'A': [[0, 1], [0, -1]], 'B': [[0, 0], [1, 1]], 'C': [[1, 0], [0, 0]]},
    ]
    
    terms = []
    
    def add_strassen_2x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """Add 7 Strassen terms for a single 2x2x2 block multiplication."""
        for t in strassen_2x2_terms:
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            for i in range(2):
                for j in range(2):
                    u[(a_r0 + i) * m + (a_c0 + j)] = t['A'][i][j]
                    v[(b_r0 + i) * p + (b_c0 + j)] = t['B'][i][j]
                    w[(c_c0 + j) * n + (c_r0 + i)] = t['C'][i][j]
            
            terms.append((u, v, w))
    
    def add_combined_product(a_blocks, b_blocks, c_blocks):
        """
        Add 7 Strassen terms for product of linear combinations of 2x2 blocks.
        a_blocks: list of (row, col, coeff) for A
        b_blocks: list of (row, col, coeff) for B  
        c_blocks: list of (row, col, coeff) for C
        """
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
                    for (r, c, coeff) in c_blocks:
                        w[(c + j) * n + (r + i)] += coeff * t['C'][i][j]
            
            terms.append((u, v, w))
    
    # 4x4x4 part using Strassen: 7 products of 2x2 matrices, each with 7 terms = 49 terms
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    add_combined_product(
        [(0, 0, 1), (2, 2, 1)],
        [(0, 0, 1), (2, 2, 1)],
        [(0, 0, 1), (2, 2, 1)]
    )
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    add_combined_product(
        [(2, 0, 1), (2, 2, 1)],
        [(0, 0, 1)],
        [(2, 0, 1), (2, 2, -1)]
    )
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    add_combined_product(
        [(0, 0, 1)],
        [(0, 2, 1), (2, 2, -1)],
        [(0, 2, 1), (2, 2, 1)]
    )
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    add_combined_product(
        [(2, 2, 1)],
        [(2, 0, 1), (0, 0, -1)],
        [(0, 0, 1), (2, 0, 1)]
    )
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    add_combined_product(
        [(0, 0, 1), (0, 2, 1)],
        [(2, 2, 1)],
        [(0, 0, -1), (0, 2, 1)]
    )
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    add_combined_product(
        [(2, 0, 1), (0, 0, -1)],
        [(0, 0, 1), (0, 2, 1)],
        [(2, 2, 1)]
    )
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    add_combined_product(
        [(0, 2, 1), (2, 2, -1)],
        [(2, 0, 1), (2, 2, 1)],
        [(0, 0, 1)]
    )
    
    # 4x4x2 part: 4 separate 2x2x2 multiplications, 4 * 7 = 28 terms
    # C_top = A00*B_top + A01*B_bottom
    add_strassen_2x2(0, 0, 0, 4, 0, 4)  # A00 * B_top -> C_top
    add_strassen_2x2(0, 2, 2, 4, 0, 4)  # A01 * B_bottom -> C_top
    
    # C_bottom = A10*B_top + A11*B_bottom
    add_strassen_2x2(2, 0, 0, 4, 2, 4)  # A10 * B_top -> C_bottom
    add_strassen_2x2(2, 2, 2, 4, 2, 4)  # A11 * B_bottom -> C_bottom
    
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
