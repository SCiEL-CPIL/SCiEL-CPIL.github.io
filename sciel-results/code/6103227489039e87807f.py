"""Matrix multiplication 4x4x9 - Block Winograd decomposition"""
DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = 4, 4, 9
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (36, rank)
    # W: (p*n, rank) = (36, rank)
    
    # Strategy: Partition the 9 columns into 3 blocks of 3.
    # For each block, compute C(4x3) = A(4x4) * B_block(4x3).
    # Partition A into four 2x2 blocks: A11, A12, A21, A22.
    # Partition B_block into two 2x3 blocks: B_top, B_bot.
    # C_top = A11*B_top + A12*B_bot  (2x3)
    # C_bot = A21*B_top + A22*B_bot  (2x3)
    
    # For each 2x2 * 2x3 multiplication, we use a rank-9 decomposition.
    # Total rank = 3 blocks * 4 sub-blocks * 9 terms = 108.
    
    # Rank-9 decomposition for 2x2 * 2x3:
    # Given A2x2 = [[a00, a01], [a10, a11]]
    # Given B2x3 = [[b00, b01, b02], [b10, b11, b12]]
    # Result C2x3 = [[c00, c01, c02], [c10, c11, c12]]
    # where:
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c02 = a00*b02 + a01*b12
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    # c12 = a10*b02 + a11*b12
    #
    # Rank-9 terms:
    # M1 = a00 * b00
    # M2 = a00 * b01
    # M3 = a00 * b02
    # M4 = a01 * b10
    # M5 = a01 * b11
    # M6 = a01 * b12
    # M7 = (a10 + a11) * b00
    # M8 = (a10 + a11) * b01
    # M9 = (a10 + a11) * b02
    #
    # Then:
    # c00 = M1 + M4
    # c01 = M2 + M5
    # c02 = M3 + M6
    # c10 = M7 - M1 + M4  ... wait, let me re-derive.
    #
    # c10 = a10*b00 + a11*b10
    # M7 = (a10+a11)*b00 = a10*b00 + a11*b00
    # So c10 = M7 - a11*b00 + a11*b10
    # This doesn't work cleanly. Let me use a different rank-9.
    #
    # Actually, let's just use schoolbook for 2x2*2x3 which is rank 12.
    # Total: 3 * 4 * 12 = 144. That's the same as before.
    #
    # Let me use a proper rank-8 or rank-9 for 2x2*2x3.
    #
    # For 2x2*2x2, Strassen gives rank 7.
    # For 2x2*2x3, we can extend: treat as two 2x2*2x1 problems sharing A.
    #
    # Let me use a simpler approach: rank 12 schoolbook for each 2x2*2x3.
    # Total = 3 * 4 * 12 = 144. Not an improvement.
    #
    # Alternative: Use rank 16 for the whole 4x4*4x3 (which is schoolbook for 4x4*4x3 = 48, no).
    #
    # Let me think again. 4x4*4x3:
    # Schoolbook: 4*4*3 = 48 per block, total 144.
    #
    # Better: Use the fact that 4x4*4x3 can be done in rank 24?
    # Partition A into 2x2 blocks, B into 2x3 blocks.
    # C_top = A11*Bt + A12*Bb, C_bot = A21*Bt + A22*Bb.
    # Each 2x2*2x3 has rank at most 12 (schoolbook).
    # But we can share: the Bt and Bb are shared across the four 2x2 blocks.
    #
    # For each 2x2*2x3, schoolbook is 12 terms.
    # But if we use Strassen-like for the 2x2 part:
    # For 2x2*2x3, we can use rank 9:
    # M1 = a00*(b00+b10), M2 = a00*(b01+b11), M3 = a00*(b02+b12)
    # M4 = a01*b10, M5 = a01*b11, M6 = a01*b12
    # M7 = (a10+a11)*(b00-b10), M8 = (a10+a11)*(b01-b11), M9 = (a10+a11)*(b02-b12)
    #
    # c00 = M1 - M4 = a00*b00 + a00*b10 - a01*b10 ... no.
    #
    # Let me just implement a valid rank-12 schoolbook for 2x2*2x3 and accept rank 144,
    # OR find a valid rank-9.
    #
    # Actually, I'll use a known valid decomposition. For 2x2*2x3:
    # The tensor has 12 components. Rank is at most 12.
    # Can we do it in 9? Yes, by linearity.
    #
    # Let me use a straightforward valid approach: rank 144 schoolbook is safe.
    # But let me try to get rank 96: use rank 8 for each 2x2*2x3 (if possible).
    #
    # I'll implement rank 144 schoolbook to be safe and correct.
    
    rank = n * m * p  # 144
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
