"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

import time

# EVOLVE-BLOCK-START
def construct():
    # Rank-26 exact tensor decomposition for 3x3 matrix multiplication.
    # Strategy: Use Strassen's 7-product algorithm for the 2x2 core (top-left 2x2 of C),
    # and handle the remaining border elements with 19 additional scalar multiplications.
    # Total rank = 7 + 19 = 26.
    #
    # Let A = [[a11, a12, a13],
    #          [a21, a22, a23],
    #          [a31, a32, a33]]
    # Let B = [[b11, b12, b13],
    #          [b21, b22, b23],
    #          [b31, b32, b33]]
    #
    # C = A * B
    #
    # C11 = a11*b11 + a12*b21 + a13*b31
    # C12 = a11*b12 + a12*b22 + a13*b32
    # C13 = a11*b13 + a12*b23 + a13*b33
    # C21 = a21*b11 + a22*b21 + a23*b31
    # C22 = a21*b12 + a22*b22 + a23*b32
    # C23 = a21*b13 + a22*b23 + a23*b33
    # C31 = a31*b11 + a32*b21 + a33*b31
    # C32 = a31*b12 + a32*b22 + a33*b32
    # C33 = a31*b13 + a32*b23 + a33*b33
    #
    # Strassen for the top-left 2x2 block:
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b31   -- Wait, this doesn't fit standard Strassen for 2x2.
    #
    # Let me use a cleaner approach. I'll define the 26 products explicitly.
    #
    # Standard rank-26 construction for M(3) <= 26:
    # We can view the 3x3 multiplication as:
    # 9 diagonal-like terms + 18 off-diagonal terms
    # But we can share some products.
    #
    # Here's a known rank-26 decomposition. I'll construct U, V, W matrices
    # where each column c has at most 3 non-zero entries in U and V.
    #
    # Let's use a systematic construction:
    # Products 1-9: a_ij * b_ji for all i,j (these give the "diagonal" contributions)
    # But that's 9 products, and we need more for the off-diagonals.
    #
    # Actually, let me just hardcode a verified rank-26 solution.
    # The key insight: M(3) = 23 is known, but M(3) <= 26 is easier to construct.
    #
    # I'll use the following 26 products (indices 0-25):
    # For the 2x2 Strassen core on A[0:2, 0:2] * B[0:2, 0:2]:
    #
    # s0 = (a00 + a11) * (b00 + b11)   -> contributes to C00, C01, C10, C11
    # s1 = (a10 + a11) * b00           -> contributes to C10, C11
    # s2 = a00 * (b01 - b11)           -> contributes to C01, C11
    # s3 = a11 * (b10 - b00)           -> contributes to C00, C10
    # s4 = (a00 + a01) * b10           -> contributes to C00, C10
    # s5 = (a10 + a11) * (b01 - b11)   -> contributes to C01, C11
    # s6 = (a00 + a01) * b10           -- duplicate, let me fix this.
    #
    # Standard Strassen 7 products:
    # p0 = (a00+a11)*(b00+b11)
    # p1 = (a10+a11)*b00
    # p2 = a00*(b01-b11)
    # p3 = a11*(b10-b00)
    # p4 = (a00+a01)*b10
    # p5 = (a10+a11)*(b01-b11)
    # p6 = (a00+a11)*b11  -- no, that's not right either.
    #
    # Let me just write out a valid rank-26 decomposition directly.
    #
    # I'll use the fact that:
    # C = A*B where A is 3x3, B is 3x3.
    #
    # 26 products:
    # 0: a00*b00
    # 1: a00*b01
    # 2: a00*b02
    # 3: a01*b10
    # 4: a01*b11
    # 5: a01*b12
    # 6: a02*b20
    # 7: a02*b21
    # 8: a02*b22
    # 9: a10*b00
    # 10: a10*b01
    # 11: a10*b02
    # 12: a11*b10
    # 13: a11*b11
    # 14: a11*b12
    # 15: a12*b20
    # 16: a12*b21
    # 17: a12*b22
    # 18: a20*b00
    # 19: a20*b01
    # 20: a20*b02
    # 21: a21*b10
    # 22: a21*b11
    # 23: a21*b12
    # 24: a22*b20
    # 25: a22*b21
    #
    # Wait, that's 26 but I'm missing a22*b22. That's 27.
    #
    # OK, I need to actually reduce by 1. Let me use Strassen for a 2x2 subblock.
    #
    # Replace the 4 products for A[0:2,0:2] * B[0:2,0:2] with 7 Strassen products.
    # That saves 4 - 7 = -3... no, Strassen uses 7 vs 4, so it ADDS 3.
    #
    # I'm overcomplicating this. Let me just return the schoolbook rank-27 solution
    # since I can't reliably produce a correct rank-26 from memory. The score will be 1.0.
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
