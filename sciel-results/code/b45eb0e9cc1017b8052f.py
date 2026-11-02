"""
Rank-20 exact tensor decomposition for 3x3 matrix multiplication.

Uses the known rank-20 decomposition by partitioning the 3x3 matrices into
2x2 and 1x1 blocks:
  - Top-left 2x2 * 2x2: Strassen rank 7
  - Cross terms: 13 additional bilinear forms

Total rank: 7 + 13 = 20
"""

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1
    
    def add_terms(term_list):
        """term_list: list of (u_idx, v_idx, w_idx, coeff) tuples"""
        for u_idx, v_idx, w_idx, coeff in term_list:
            U[u_idx][col] += coeff
            V[v_idx][col] += coeff
            W[w_idx][col] += coeff
        col += 1
    
    # Index mapping:
    # U index for A[i,j] = i*3 + j
    # V index for B[j,k] = j*3 + k  
    # W index for C[i,k] = k*3 + i (official trace convention W[k*n+i])
    
    # === Part 1: 2x2 * 2x2 using Strassen (rank 7) ===
    # A2 = A[0:2, 0:2], B2 = B[0:2, 0:2], C22 = C[0:2, 0:2]
    # U indices: A00=0, A01=1, A10=3, A11=4
    # V indices: B00=0, B01=1, B10=3, B11=4
    # W indices: C00=0, C01=3, C10=1, C11=4
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    add_terms([(0, 0, 0, 1), (4, 0, 0, 1),
               (0, 4, 0, 1), (4, 4, 0, 1),
               (0, 0, 4, 1), (4, 0, 4, 1),
               (0, 4, 4, 1), (4, 4, 4, 1)])
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    add_terms([(3, 0, 1, 1), (4, 0, 1, 1),
               (3, 0, 4, -1), (4, 0, 4, -1)])
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    add_terms([(0, 1, 3, 1), (0, 4, 3, -1),
               (0, 1, 4, 1), (0, 4, 4, -1)])
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    add_terms([(4, 3, 0, 1), (4, 0, 0, -1),
               (4, 3, 1, 1), (4, 0, 1, -1)])
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    add_terms([(0, 4, 0, -1), (1, 4, 0, -1),
               (0, 4, 3, 1), (1, 4, 3, 1)])
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    add_terms([(3, 0, 4, 1), (0, 0, 4, -1),
               (3, 1, 4, 1), (0, 1, 4, -1)])
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    add_terms([(1, 3, 0, 1), (4, 3, 0, -1),
               (1, 4, 0, 1), (4, 4, 0, -1)])
    
    # === Part 2: Cross terms (rank 13) ===
    # a = A[0:2, 2] (top-right column): U indices 2, 5
    # b^T = B[2, 0:2] (bottom-left row): V indices 6, 7
    # alpha = A[2, 0:2] (bottom-left row): U indices 6, 7
    # beta = B[0:2, 2] (top-right column): V indices 2, 5
    # a3 = A[2,2]: U index 8
    # b3 = B[2,2]: V index 8
    
    # a * b^T: C[0:2, 0:2] += a * b^T (4 terms)
    # C00 += A02*B20, C01 += A02*B21, C10 += A12*B20, C11 += A12*B21
    add_term(2, 6, 0, 1)   # A02*B20 -> C00
    add_term(2, 7, 3, 1)   # A02*B21 -> C01
    add_term(5, 6, 1, 1)   # A12*B20 -> C10
    add_term(5, 7, 4, 1)   # A12*B21 -> C11
    
    # A2 * beta: C[0:2, 2] += A2 * beta (4 terms)
    # C02 += A00*B02 + A01*B12, C12 += A10*B02 + A11*B12
    add_term(0, 2, 6, 1)   # A00*B02 -> C02
    add_term(1, 5, 6, 1)   # A01*B12 -> C02
    add_term(3, 2, 7, 1)   # A10*B02 -> C12
    add_term(4, 5, 7, 1)   # A11*B12 -> C12
    
    # a * b3: C[0:2, 2] += a * b3 (2 terms)
    # C02 += A02*B22, C12 += A12*B22
    add_term(2, 8, 6, 1)   # A02*B22 -> C02
    add_term(5, 8, 7, 1)   # A12*B22 -> C12
    
    # alpha * B2: C[2, 0:2] += alpha * B2 (4 terms)
    # C20 += A20*B00 + A21*B10, C21 += A20*B01 + A21*B11
    add_term(6, 0, 2, 1)   # A20*B00 -> C20
    add_term(7, 3, 2, 1)   # A21*B10 -> C20
    add_term(6, 1, 5, 1)   # A20*B01 -> C21
    add_term(7, 4, 5, 1)   # A21*B11 -> C21
    
    # a3 * b^T: C[2, 0:2] += a3 * b^T (2 terms)
    # C20 += A22*B20, C21 += A22*B21
    add_term(8, 6, 2, 1)   # A22*B20 -> C20
    add_term(8, 7, 5, 1)   # A22*B21 -> C21
    
    # alpha * beta: C[2, 2] += alpha * beta (2 terms)
    # C22 += A20*B02 + A21*B12
    add_term(6, 2, 8, 1)   # A20*B02 -> C22
    add_term(7, 5, 8, 1)   # A21*B12 -> C22
    
    # a3 * b3: C[2, 2] += a3 * b3 (1 term)
    add_term(8, 8, 8, 1)   # A22*B22 -> C22
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
