import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 140
    n, m, p = DIMENSIONS  # all 6

    # Strassen 2x2 coefficients
    # U, V, W terms for 2x2 matrix multiplication
    # Each term is a list of (row, col, coeff)
    strassen_U = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)],
    ]
    strassen_V = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
    ]
    strassen_W = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    rank = 182
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_strassen_2x2(row_off_a, col_off_a, row_off_b, col_off_b, row_off_c, col_off_c):
        nonlocal col
        for t in range(7):
            for (li, lk, coeff) in strassen_U[t]:
                gi = row_off_a + li
                gk = col_off_a + lk
                U[gi * m + gk][col] += coeff
            for (lk, lj, coeff) in strassen_V[t]:
                gk = row_off_b + lk
                gj = col_off_b + lj
                V[gk * p + gj][col] += coeff
            for (li, lj, coeff) in strassen_W[t]:
                gi = row_off_c + li
                gj = col_off_c + lj
                W[gj * n + gi][col] += coeff
            col += 1

    def add_strassen_4x4(row_off_a, col_off_a, row_off_b, col_off_b, row_off_c, col_off_c):
        # 4x4 Strassen: 7 block multiplications of 2x2 blocks
        # The blocks are at offsets (0,0), (0,2), (2,0), (2,2) relative to the 4x4 start
        # Strassen terms for 2x2 block matrices:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to add 7 2x2 Strassen decompositions.
        # But the "matrices" being multiplied are 2x2 matrices of 2x2 blocks.
        # So we need to apply the Strassen coefficients to the block indices.
        
        # Block indices:
        # A: A00 (0,0), A01 (0,2), A10 (2,0), A11 (2,2)
        # B: B00 (0,0), B01 (0,2), B10 (2,0), B11 (2,2)
        # C: C00 (0,0), C01 (0,2), C10 (2,0), C11 (2,2)
        
        # Term 1: (A00+A11)(B00+B11)
        # A part: A00 + A11. B part: B00 + B11.
        # This is a single 2x2 Strassen decomposition where the "matrices" are sums?
        # No, the Strassen algorithm computes 7 products.
        # Each product is of the form (Sum of A blocks) * (Sum of B blocks).
        # We can distribute the coefficients.
        
        # Instead of implementing the full Strassen recursion on blocks,
        # we can just note that R(4) via Strassen is 7 * R(2) = 49.
        # We can implement this by calling add_strassen_2x2 7 times?
        # No, that would be 7 * 7 = 49 rank.
        # But the structure is different.
        
        # Let's use the explicit 7 terms for 4x4 Strassen.
        # Each term involves a product of two 2x2 block matrices.
        # We need to decompose each such product.
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # A_M1 = A00 + A11. B_M1 = B00 + B11.
        # We add a 2x2 Strassen decomposition for A_M1 * B_M1 -> C_M1.
        # C_M1 contributes to C00 and C11.
        
        # This is getting complex to map manually.
        # I will use the 3x3 grid of 2x2 blocks (Rank 189) to be safe.
        pass

    # Fallback to 189 if 182 implementation is too complex/risky
    # I will stick to 189 for correctness.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    for bi in range(3):
        for bk in range(3):
            for bj in range(3):
                add_strassen_2x2(2*bi, 2*bk, 2*bk, 2*bj, 2*bi, 2*bj)
                
    if _time.time() > _deadline:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
