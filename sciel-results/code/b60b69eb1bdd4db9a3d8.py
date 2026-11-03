"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 5
    
    # Use Strassen's algorithm for the 3x3 * 3x3 part (rank 23)
    # and standard multiplication for the remaining 2 columns (rank 17)
    # Total rank: 23 + 17 = 40
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]  # 9 x 40
    V = [[0] * rank for _ in range(m * p)]  # 15 x 40
    W = [[0] * rank for _ in range(p * n)]  # 15 x 40
    
    # Strassen decomposition for 3x3 * 3x3
    # The 23 terms of Strassen's algorithm for 3x3 matrices
    # We need to embed this into our 3x3 * 3x5 problem
    
    # For the first 3 columns (k=0,1,2), use Strassen's 23 terms
    # For columns 3,4, use standard terms: 3*3*2 = 18 terms
    # But we can share some structure. Let's use 23 + 17 = 40.
    
    # Strassen's 23 terms for 3x3 * 3x3:
    # The 3x3 matrix is partitioned into 2x2 blocks with overlap.
    # Actually, for 3x3, Strassen uses a specific set of 23 bilinear forms.
    
    # Let me use a known rank-23 decomposition for 3x3*3x3.
    # The decomposition uses the following 23 terms:
    
    strassen_terms = [
        # Each term is (U_indices, V_indices, W_indices)
        # U: 9 entries (i*m+j), V: 9 entries (j*p+k) for k=0,1,2, W: 9 entries (k*n+i)
        
        # Term 0: M0 = A00 * B00
        ((0,), (0,), (0,)),
        # Term 1: M1 = A00 * (B01 + B11) 
        ((0,), (1, 5), (1, 4)),
        # Term 2: M2 = (A10 + A20) * B00
        ((3, 6), (0,), (3, 6)),
        # Term 3: M3 = A22 * B22
        ((8,), (8,), (8,)),
        # Term 4: M4 = (A00 + A22) * (B00 + B22)
        ((0, 8), (0, 8), (0, 8)),
        # Term 5: M5 = (A02 + A22) * B20
        ((2, 8), (6,), (2, 5)),
        # Term 6: M6 = A00 * B20
        ((0,), (6,), (2,)),
        # Term 7: M7 = (A20 - A00) * (B00 + B02)
        ((3, 0), (0, 2), (3, 5)),
        # Term 8: M8 = (A02 + A22) * B20 - ... this is getting complex
        
        # Let me use a cleaner approach: explicit rank-40 construction
    ]
    
    # I'll use a direct construction that's verifiable.
    # For (3,3,5), a rank-40 construction exists.
    # Let me use the following approach:
    # - 27 standard terms for 3x3*3x3 (k=0,1,2)
    # - 13 terms for the remaining 2 columns (k=3,4)
    # Total: 40
    
    # But 27+13=40, and we can potentially do better.
    # For now, let me implement a valid rank-45 construction and verify it.
    
    # Actually, let me implement a rank-40 construction using a known method.
    # The key insight: we can reuse Strassen terms for the 3x3 part.
    
    # Let me use a simpler valid construction: rank 45 (schoolbook)
    # This is guaranteed to be correct.
    
    rank = 45
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
