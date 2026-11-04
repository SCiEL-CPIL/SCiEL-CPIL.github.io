"""Matrix multiplication 2x3x8 with rank-minimal decomposition."""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Decompose into 4 independent 2x3x2 sub-problems
    # Each 2x3x2 uses rank 6 via Winograd/Strassen-like method
    # Total rank = 4 * 6 = 24
    
    rank_per_block = 6
    num_blocks = p // 2  # 4 blocks
    rank = rank_per_block * num_blocks  # 24
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 24
    V = [[0] * rank for _ in range(m * p)]  # 24 x 24
    W = [[0] * rank for _ in range(p * n)]  # 16 x 24
    
    # For each block (column pair of B), define the rank-6 decomposition
    # A is 2x3: indices (i,j) with i in {0,1}, j in {0,1,2}
    # B_group is 3x2: indices (j,k) with j in {0,1,2}, k in {0,1}
    # C is 2x2: indices (i,k) with i in {0,1}, k in {0,1}
    
    # Rank-6 decomposition for 2x3x2:
    # M1 = (a00+a10)*(b00+b01)
    # M2 = (a01+a11)*(b10+b11)
    # M3 = (a02+a12)*(b20+b21)
    # M4 = (a00-a10)*b00
    # M5 = (a01-a11)*b10
    # M6 = (a02-a12)*b20
    #
    # Reconstruction:
    # C[0,0] = (M1 + M4)/2 + M5 + M6 - (M2 + M5)/2 - (M3 + M6)/2 + ... 
    # This needs careful derivation. Let me use a verified scheme.
    
    # Verified rank-6 scheme for 2x3x2:
    # Let u1 = a00+a10, u2 = a01+a11, u3 = a02+a12
    #     u4 = a00-a10, u5 = a01-a11, u6 = a02-a12
    # Let v1 = b00+b01, v2 = b10+b11, v3 = b20+b21
    #     v4 = b00, v5 = b10, v6 = b20
    # M1 = u1*v1, M2 = u2*v2, M3 = u3*v3
    # M4 = u4*v4, M5 = u5*v5, M6 = u6*v6
    #
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    #
    # C[0,0] + C[1,0] = (a00+a10)*b00 + (a01+a11)*b10 + (a02+a12)*b20
    # C[0,1] + C[1,1] = (a00+a10)*b01 + (a01+a11)*b11 + (a02+a12)*b21
    #
    # M1 = (a00+a10)*b00 + (a00+a10)*b01
    # M2 = (a01+a11)*b10 + (a01+a11)*b11
    # M3 = (a02+a12)*b20 + (a02+a12)*b21
    # M4 = (a00-a10)*b00
    # M5 = (a01-a11)*b10
    # M6 = (a02-a12)*b20
    #
    # So: (C[0,0]+C[1,0]) = M4 + M5 + M6 + (a00+a10)*b00 - (a00-a10)*b00 ... 
    # Actually: (a00+a10)*b00 = (M4 + (a00+a10)*b00 - (a00-a10)*b00)/... 
    
    # Let me just use: 
    # C[0,0] = (M1 + M4)/2 + (M2 + M5)/2 + (M3 + M6)/2 - (C[0,1] + C[1,0])/2
    # This is circular. 
    
    # Alternative: use the fact that we can express each C entry as a linear combo of M's.
    # Let me solve for the coefficients.
    
    # For now, use schoolbook per block (rank 12) for correctness, total rank 48.
    # But try to find a valid rank-6 decomposition.
    
    # Actually, let's use a different valid rank-6 decomposition:
    # M1 = (a00+a10)*(b00+b01)
    # M2 = (a01+a11)*(b10+b11)
    # M3 = (a02+a12)*(b20+b21)
    # M4 = a00*b01
    # M5 = a01*b11
    # M6 = a02*b21
    #
    # C[0,0] = a00*b00 + a01*b10 + a02*b20 = (M1 - M4) - a10*b00 - a10*b01 + ... 
    
    # I'll use the straightforward schoolbook for each 2x3x2 block (rank 12 per block, 48 total)
    # since deriving a valid rank-6 is complex.
    
    # Schoolbook for each 2x3x2 block:
    for block in range(num_blocks):
        base_col = block * rank_per_block
        # For this block, B columns are at positions 2*block and 2*block+1
        for i in range(n):
            for j in range(m):
                for k in range(2):
                    col = base_col + (i * m + j) * 2 + k  # This won't work with rank 6
                    # Use schoolbook: rank 12 per block
                    pass
    
    # Fall back to full schoolbook
    rank = n * m * p  # 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
