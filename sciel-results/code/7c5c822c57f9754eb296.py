"""Matrix multiplication 2x2 x 2x12 via Kronecker product of Strassen 2x2x2 and 1x1x6."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time

def construct():
    """Rank-42 decomposition: Kronecker product of Strassen (2x2x2, rank 7) and identity (1x1x6, rank 6).
    
    T_2x2x12 = T_2x2x2 ⊗ T_1x1x6
    Rank <= 7 * 6 = 42
    
    Indexing:
      U[i*m+j, r]: A[i][j]
      V[j*p+k, r]: B[j][k]  
      W[k*n+i, r]: C[i][k] (trace convention)
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 2, 12
    num_pairs = p // 2  # 6
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]    # 4 x 42
    V = [[0] * rank for _ in range(m * p)]    # 24 x 42
    W = [[0] * rank for _ in range(p * n)]    # 24 x 42
    
    # Strassen 7 products for 2x2 x 2x2:
    # Products s=0..6: U_coeffs (on A), V_coeffs (on B), W_coeffs (on C)
    # A indices: (i,j), B indices: (j,k), C indices: (i,k)
    strassen = [
        # s=0: (A00+A11)(B00+B11) -> C00 += , C11 +=
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # s=1: (A10+A11)B00 -> C10 += , C11 -=
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # s=2: A00(B01-B11) -> C01 += , C11 +=
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # s=3: A11(B10-B00) -> C00 += , C10 +=
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # s=4: (A00+A01)B11 -> C00 -= , C01 +=
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # s=5: (A10-A00)(B00+B01) -> C11 +=
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # s=6: (A01-A11)(B10+B11) -> C00 +=
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    col = 0
    for t in range(num_pairs):
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
        base_col = 2 * t
        for s in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen[s]
            for (ai, aj, coeff) in u_coeffs:
                U[ai * m + aj][col] += coeff
            for (bj, bk_rel, coeff) in v_coeffs:
                V[bj * p + base_col + bk_rel][col] += coeff
            for (ci, ck_rel, coeff) in w_coeffs:
                W[(base_col + ck_rel) * n + ci][col] += coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
