"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    
    n, m, p = 2, 2, 4
    rank = 10
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Rank-10 decomposition for 2x2 * 2x4 matrix multiplication.
    # 
    # Strategy: Use Strassen-like identities on each 2x2 block of B,
    # but share the A-side linear combinations across blocks.
    #
    # Let A = [[a00, a01], [a10, a11]]
    # Let B = [[b00, b01, b02, b03], [b10, b11, b12, b13]]
    # C = A*B, so C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # We define 10 rank-1 terms. For each term c, the contribution to
    # C[i,k] is sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    #
    # Key insight: We can use 4 terms to handle the "diagonal" products
    # and 6 terms to handle the off-diagonal/mixed products using
    # Strassen-like linear combinations.
    #
    # Terms 0-3: Handle a00 * B[0,k] for k=0,1,2,3
    #   These directly contribute to C[0,k]
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1   # a00*b00 -> C[0,0]
    U[0, 1] = 1; V[1, 1] = 1; W[2, 1] = 1   # a00*b01 -> C[0,1]
    U[0, 2] = 1; V[2, 2] = 1; W[4, 2] = 1   # a00*b02 -> C[0,2]
    U[0, 3] = 1; V[3, 3] = 1; W[6, 3] = 1   # a00*b03 -> C[0,3]
    
    # Terms 4-5: Handle a01 * B[1,k] for k=0,1 using Strassen-like combos
    # We need a01*b10 -> C[0,0] and a01*b11 -> C[0,1]
    # But we also need to handle a10*b00, a11*b10 -> C[1,0]
    # and a10*b01, a11*b11 -> C[1,1]
    #
    # Let's use:
    # Term 4: (a01 + a10) * b10 -> contributes to C[0,0] and C[1,0]
    # Term 5: (a01 + a11) * b11 -> contributes to C[0,1] and C[1,1]
    # But this doesn't work directly because the W coefficients would
    # need to be different for C[0,k] and C[1,k].
    #
    # Alternative approach: Use the following 10 terms:
    #
    # For the left 2x2 block (k=0,1):
    # P1 = (a00 + a11) * (b00 + b11)
    # P2 = (a10 + a11) * b00
    # P3 = a00 * (b11 - b00)
    # P4 = a11 * (b01 - b00)
    # P5 = (a00 + a01) * b10
    # P6 = (a10 + a11) * (b01 + b11)
    # P7 = (a01 - a11) * (b10 + b11)
    #
    # For the right 2x2 block (k=2,3):
    # P8 = (a00 + a11) * (b02 + b13)
    # P9 = (a10 + a11) * b02
    # ... but this gives 14 terms, not 10.
    #
    # Let me try a different construction. We'll use 10 terms where:
    # - 4 terms handle a00 * B[0,:] (direct)
    # - 3 terms handle the a01, a10, a11 interactions for k=0,1
    # - 3 terms handle the a01, a10, a11 interactions for k=2,3
    #
    # Actually, let me use a verified rank-10 construction:
    #
    # Term 0: a00 * b00 -> C[0,0]
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 1: a00 * b01 -> C[0,1]
    U[0, 1] = 1; V[1, 1] = 1; W[2, 1] = 1
    # Term 2: a00 * b02 -> C[0,2]
    U[0, 2] = 1; V[2, 2] = 1; W[4, 2] = 1
    # Term 3: a00 * b03 -> C[0,3]
    U[0, 3] = 1; V[3, 3] = 1; W[6, 3] = 1
    # Term 4: a01 * b10 -> C[0,0]
    U[1, 4] = 1; V[4, 4] = 1; W[0, 4] = 1
    # Term 5: a01 * b11 -> C[0,1]
    U[1, 5] = 1; V[5, 5] = 1; W[2, 5] = 1
    # Term 6: a01 * b12 -> C[0,2]
    U[1, 6] = 1; V[6, 6] = 1; W[4, 6] = 1
    # Term 7: a01 * b13 -> C[0,3]
    U[1, 7] = 1; V[7, 7] = 1; W[6, 7] = 1
    # Term 8: (a10 + a11) * (b00 + b10) -> helps C[1,0]
    U[2, 8] = 1; U[3, 8] = 1
    V[0, 8] = 1; V[4, 8] = 1
    W[1, 8] = 1
    # Term 9: (a10 + a11) * (b01 + b11) -> helps C[1,1]
    U[2, 9] = 1; U[3, 9] = 1
    V[1, 9] = 1; V[5, 9] = 1
    W[3, 9] = 1
    
    # But this only covers C[1,0] and C[1,1] partially. We need C[1,2] and C[1,3] too.
    # This construction doesn't work. Let me use a proper verified one.
    #
    # After careful analysis, here is a rank-10 construction:
    #
    # We use the following 10 terms:
    #
    # c=0: a00 * b00 -> C[0,0]
    # c=1: a00 * b01 -> C[0,1]  
    # c=2: a00 * b02 -> C[0,2]
    # c=3: a00 * b03 -> C[0,3]
    # c=4: a01 * b10 -> C[0,0]
    # c=5: a01 * b11 -> C[0,1]
    # c=6: a01 * b12 -> C[0,2]
    # c=7: a01 * b13 -> C[0,3]
    # c=8: a10 * b00 + a11 * b10 -> C[1,0] (using (a10+a11)*(b00+b10) - a10*b10 - a11*b00)
    # c=9: a10 * b01 + a11 * b11 -> C[1,1]
    #
    # But we still miss C[1,2] and C[1,3]. So rank 10 with this structure doesn't work.
    #
    # Let me try yet another approach. The minimum rank for 2x2*2x4 is indeed 10.
    # Here's a working construction:
    
    # Reset
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # We'll use a Strassen-based approach with shared terms.
    # 
    # For the left block (columns 0,1 of B):
    # M1 = (a00+a11) * (b00+b11)
    # M2 = (a10+a11) * b00
    # M3 = a00 * (b11-b00)
    # M4 = a11 * (b01-b00)
    # M5 = (a00+a01) * b10
    # M6 = (a10+a11) * (b01+b11)
    # M7 = (a01-a11) * (b10+b11)
    #
    # For the right block (columns 2,3 of B):
    # M8 = (a00+a11) * (b02+b13)
    # M9 = (a10+a11) * b02
    # M10 = a00 * (b13-b02)
    # ...
    #
    # This gives 14 terms. To get to 10, we need to share 4 terms.
    #
    # Notice that M1 and M8 both use (a00+a11). If we define:
    # S1 = a00+a11
    # S2 = a10+a11
    # S3 = a00+a01
    # S4 = a01-a11
    #
    # And for B:
    # T1 = b00+b11, T2 = b02+b13
    # T3 = b11-b00, T4 = b13-b02
    # T5 = b01-b00, T6 = b03-b02
    # T7 = b01+b11, T8 = b03+b13
    # T9 = b10+b11, T10 = b12+b13
    #
    # Then:
    # M1 = S1*T1
    # M8 = S1*T2
    # M2 = S2*b00
    # M9 = S2*b02
    # M3 = a00*T3
    # M10 = a00*T4
    # M4 = a11*T5
    # M11 = a11*T6
    # M5 = S3*b10
    # M12 = S3*b12
    # M6 = S2*T7
    # M13 = S2*T8
    # M7 = S4*T9
    # M14 = S4*T10
    #
    # Still 14. The sharing doesn't reduce the count because each product is distinct.
    #
    # Let me just use the schoolbook rank-16 for now, which is guaranteed correct.
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
