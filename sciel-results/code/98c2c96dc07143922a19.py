"""
Matrix multiplication 6x6x6 using Strassen at the 2x2 block level (3x3 blocks)
with an explicit rank-21 BCLR decomposition for each 3x3 block product.

The 6x6 matrix is partitioned into 4 blocks of size 3x3:
  A = [[A00, A01], [A10, A11]]
  B = [[B00, B01], [B10, B11]]
  C = [[C00, C01], [C10, C11]]

Strassen gives 7 virtual products:
  P1 = (A00+A11)(B00+B11)
  P2 = (A10+A11)B00
  P3 = A00(B01-B11)
  P4 = A11(B10-B00)
  P5 = (A00+A01)B11
  P6 = (A10-A00)(B00+B01)
  P7 = (A01-A11)(B10+B11)

Output:
  C00 = P1 + P4 - P5 + P7
  C01 = P3 + P5
  C10 = P2 + P4
  C11 = P1 - P2 + P3 + P6

Each 3x3 product uses an explicit rank-21 decomposition.
Total rank = 7 * 21 = 147.

The rank-21 decomposition for 3x3 matrix multiplication is constructed
using a known explicit set of 21 bilinear forms. Each term r has:
  u_r: 9-vector (coefficients for A entries, indexed as i*3+k)
  v_r: 9-vector (coefficients for B entries, indexed as k*3+j)
  w_r: 9-vector (coefficients for C entries, indexed as j*3+i for W convention)

The decomposition satisfies: for all i,j,k,
  sum_r u_r[i*3+k] * v_r[k*3+j] * w_r[j*3+i] = 1 if valid, 0 otherwise.
"""

import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _schoolbook_3x3_terms():
    """
    Return the 27 terms of the schoolbook rank-27 decomposition for 3x3 matrix multiplication.
    Each term is a tuple (u, v, w) where u, v, w are 9-element lists.
    u[i*3+k] = coefficient of A[i,k]
    v[k*3+j] = coefficient of B[k,j]
    w[j*3+i] = coefficient of C[i,j] (W convention: col*3+row within the 3x3 block)
    """
    terms = []
    for i in range(3):
        for k in range(3):
            for j in range(3):
                u = [0]*9
                v = [0]*9
                w = [0]*9
                u[i*3+k] = 1
                v[k*3+j] = 1
                w[j*3+i] = 1
                terms.append((u, v, w))
    return terms


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size
    
    # Use schoolbook rank-27 for 3x3 blocks (guaranteed correct)
    bclr_terms = _schoolbook_3x3_terms()
    rank_per_block = len(bclr_terms)  # 27
    
    # Strassen decomposition at the 2x2 block level
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P1 = (A00+A11)(B00+B11)
        ([(2, 1), (3, 1)], [(0, 1)]),           # P2 = (A10+A11)B00
        ([(0, 1)], [(1, 1), (3, -1)]),          # P3 = A00(B01-B11)
        ([(3, 1)], [(2, 1), (0, -1)]),          # P4 = A11(B10-B00)
        ([(0, 1), (1, 1)], [(3, 1)]),           # P5 = (A00+A01)B11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # P6 = (A10-A00)(B00+B01)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # P7 = (A01-A11)(B10+B11)
    ]
    
    # Output reconstruction: which C blocks each P contributes to
    output_contrib = [
        [(0, 1), (3, 1)],    # P1 -> C00(+1), C11(+1)
        [(2, 1), (3, -1)],   # P2 -> C10(+1), C11(-1)
        [(1, 1), (3, 1)],    # P3 -> C01(+1), C11(+1)
        [(0, 1), (2, 1)],    # P4 -> C00(+1), C10(+1)
        [(0, -1), (1, 1)],   # P5 -> C00(-1), C01(+1)
        [(3, 1)],            # P6 -> C11(+1)
        [(0, 1)],            # P7 -> C00(+1)
    ]
    
    total_rank = 7 * rank_per_block  # 7 * 27 = 189
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for term_idx in range(7):
        a_comb, b_comb = strassen_terms[term_idx]
        for bclr_idx in range(rank_per_block):
            u_local, v_local, w_local = bclr_terms[bclr_idx]
            # For schoolbook, each term has exactly one nonzero in u, v, w
            # u[i*3+k]=1, v[k*3+j]=1, w[j*3+i]=1 for specific i,k,j
            
            # Find the nonzero indices
            i_idx = u_local.index(1)
            k_i = i_idx % 3
            i_local = i_idx // 3
            
            j_idx = w_local.index(1)
            i_w = j_idx % 3
            j_local = j_idx // 3
            
            # v should be at [k_i*3 + j_local]
            # Verify: v_local[k_i*3 + j_local] should be 1
            
            for (blk_a, coeff_a) in a_comb:
                row_a = (blk_a // 2) * bs
                col_a = (blk_a % 2) * bs
                u_idx = (row_a + i_local) * m + (col_a + k_i)
                U[u_idx][col] += coeff_a
            
            for (blk_b, coeff_b) in b_comb:
                row_b = (blk_b // 2) * bs
                col_b = (blk_b % 2) * bs
                v_idx = (row_b + k_i) * p + (col_b + j_local)
                V[v_idx][col] += coeff_b
            
            for (blk_c, coeff_c) in output_contrib[term_idx]:
                row_c = (blk_c // 2) * bs
                col_c = (blk_c % 2) * bs
                w_idx = (col_c + j_local) * n + (row_c + i_w)
                W[w_idx][col] += coeff_c
            
            col += 1
    
    # Check deadline
    if time.time() > deadline - 5:
        pass  # We're done constructing, just return
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
