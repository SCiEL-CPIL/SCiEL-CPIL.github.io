"""Strassen-based rank-42 decomposition for 2x2 * 2x12 matrix multiplication."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time

def construct():
    """Rank-42 exact tensor decomposition for 2x2 * 2x12 matrix multiplication.
    
    Uses Strassen's 7-product algorithm on 6 independent 2-column blocks.
    
    Indexing conventions (trace convention for W):
      U[i*m+j, r]: row = i*m+j (i in 0..n-1, j in 0..m-1) for A[i][j]
      V[j*p+k, r]: row = j*p+k (j in 0..m-1, k in 0..p-1) for B[j][k]
      W[k*n+i, r]: row = k*n+i (k in 0..p-1, i in 0..n-1) for C[i][k]
    """
    n, m, p = DIMENSIONS
    num_pairs = p // 2  # 6 pairs of 2 columns
    
    rank = num_pairs * 7  # 42
    U = [[0] * rank for _ in range(n * m)]   # 4 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 24 x 42
    
    # Strassen 7 products for 2x2 * 2x2
    # Each product: (U_part, V_part) where parts are lists of ((index, coeff), ...)
    # U_part indexes into A[i][j] with i,j in {0,1}
    # V_part indexes into B[j][k] with j in {0,1}, k in {0,1} (relative to block)
    products = [
        # M1 = (A00+A11)(B00+B11)
        ([( (0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        # M2 = (A10+A11)B00
        ([( (1,0), 1), ((1,1), 1)], [((0,0), 1)]),
        # M3 = A00(B01-B11)
        ([( (0,0), 1)], [((0,1), 1), ((1,1), -1)]),
        # M4 = A11(B10-A00)
        ([( (1,1), 1)], [((1,0), 1), ((0,0), -1)]),
        # M5 = (A00+A01)B11
        ([( (0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        # M6 = (A10-A00)(B00+B01)
        ([( (1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),
        # M7 = (A01-A11)(B10+B11)
        ([( (0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)]),
    ]
    
    # Output contributions for each C[i][k] from each M_s
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    output_contribs = [
        [((0,0), 1), ((1,1), 1)],   # M1 -> C00(+1), C11(+1)
        [((1,0), 1), ((1,1), -1)],  # M2 -> C10(+1), C11(-1)
        [((0,1), 1), ((1,1), 1)],   # M3 -> C01(+1), C11(+1)
        [((0,0), 1), ((1,0), 1)],   # M4 -> C00(+1), C10(+1)
        [((0,0), -1), ((0,1), 1)],  # M5 -> C00(-1), C01(+1)
        [((1,1), -1)],              # M6 -> C11(-1)
        [((0,0), -1)],              # M7 -> C00(-1)
    ]
    
    col = 0
    for t in range(num_pairs):
        base_col = 2 * t
        for s in range(7):
            # Fill U
            for (ai, aj), coeff in products[s][0]:
                u_row = ai * m + aj
                U[u_row][col] += coeff
            
            # Fill V
            for (bj, bk_rel), coeff in products[s][1]:
                b_col = base_col + bk_rel
                v_row = bj * p + b_col
                V[v_row][col] += coeff
            
            # Fill W (trace convention: W[k*n+i, r])
            for (ci, ck_rel), coeff in output_contribs[s]:
                c_col = base_col + ck_rel
                w_row = c_col * n + ci
                W[w_row][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
