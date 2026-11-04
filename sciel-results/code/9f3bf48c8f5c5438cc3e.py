"""Matrix multiplication 2x3x8 with rank-minimal decomposition."""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Strategy: Decompose into 4 independent 2x3x2 sub-problems
    # Each 2x3x2 uses a rank-6 decomposition (Strassen-like for 2x2x2 + 2 scalar mults for the 3rd column)
    # Total rank = 4 * 6 = 24
    
    rank = 24
    n_m = n * m  # 6
    m_p = m * p  # 24
    p_n = p * n  # 16
    
    U = [[0] * rank for _ in range(n_m)]
    V = [[0] * rank for _ in range(m_p)]
    W = [[0] * rank for _ in range(p_n)]
    
    # For each block of 2 columns (k, k+1) in the 3x8 matrix:
    # We compute A (2x3) * B_block (3x2) -> C_block (2x2)
    # 
    # Let A = [[a00, a01, a02], [a10, a11, a12]]
    # Let B_block = [[b00, b01], [b10, b11], [b20, b21]]
    # C_block = [[c00, c01], [c10, c11]]
    #
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    #
    # We can decompose this into 6 multiplications using:
    # M1 = (a00+a10) * (b00+b01)   -- covers a00*b00, a00*b01, a10*b00, a10*b01
    # M2 = (a01+a11) * (b10+b11)   -- covers a01*b10, a01*b11, a11*b10, a11*b11
    # M3 = (a02+a12) * (b20+b21)   -- covers a02*b20, a02*b21, a12*b20, a12*b21
    # M4 = (a00-a10) * b00         -- helps isolate a00*b00 vs a10*b00
    # M5 = (a01-a11) * b10         -- helps isolate a01*b10 vs a11*b10
    # M6 = (a02-a12) * b20         -- helps isolate a02*b20 vs a12*b20
    #
    # Actually, let's use a cleaner Strassen-style for the 2x2x2 part (first 2 columns of B_block)
    # and 2 more for the 3rd column interactions.
    #
    # Better approach: Use the known rank-6 decomposition for 2x3x2.
    # 
    # M1 = a00 * b00
    # M2 = a00 * b01
    # M3 = a01 * b10
    # M4 = a01 * b11
    # M5 = a02 * b20
    # M6 = a02 * b21
    # That's only 6 but doesn't cover a10, a11, a12 terms.
    #
    # Let me think again. The standard approach for 2x3x2:
    # We have 4 output entries, each is a sum of 3 products.
    # Total 12 products in schoolbook.
    # 
    # A rank-6 decomposition exists if we can share products.
    # Consider:
    # M1 = (a00+a10) * b00
    # M2 = (a00+a10) * b01
    # M3 = (a01+a11) * b10
    # M4 = (a01+a11) * b11
    # M5 = (a02+a12) * b20
    # M6 = (a02+a12) * b21
    #
    # Then:
    # c00 = M1 - a10*b00 + a01*b10 + a02*b20  -- still has leftover terms
    #
    # This doesn't work directly. Let me use a different known decomposition.
    #
    # Actually, for 2x3x2, the minimum rank is 12, not 6. Let me reconsider.
    # 
    # Wait, let me check: M(2,3,2) >= M(2,2,2) = 7. And M(2,3,2) >= 2*3 = 6 (trivial).
    # But I believe M(2,3,2) = 12 actually, since the 3rd column of A doesn't interact with 1st/2nd columns of B in a shareable way.
    #
    # Let me just use schoolbook for each 2x3x2 block: 12 multiplications each, 4 blocks = 48.
    # But that's the same as the original.
    #
    # Alternative: View it as 2x3x8 directly and find a better decomposition.
    #
    # Let me try a different approach: use the fact that 8 = 2^3 and try a recursive/FFT-like approach.
    # Or use the Winograd method for 2x3x8.
    #
    # Actually, let me just implement a careful Strassen-like decomposition for the full 2x3x8.
    #
    # For now, let me return the schoolbook solution to ensure correctness, then optimize.
    
    # Schoolbook: rank = 48
    rank = n * m * p
    U = [[0] * rank for _ in range(n_m)]
    V = [[0] * rank for _ in range(m_p)]
    W = [[0] * rank for _ in range(p_n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
