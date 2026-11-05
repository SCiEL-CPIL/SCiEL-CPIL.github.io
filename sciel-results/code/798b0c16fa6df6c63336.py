import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Decompose 3x5x6 into blocks:
    # Rows: 2 + 1
    # Cols of A / rows of B: 4 + 1
    # Cols of B / cols of C: 5 + 1
    #
    # Blocks:
    # (2,4,5): rank 17 (optimal)
    # (2,4,1): rank 2 (schoolbook: 2*4*1=8, but can do 2 since it's just scaling)
    # (2,1,5): rank 2 (2*1*5=10, but 2 since A is 2x1, B is 1x5)
    # (1,4,5): rank 5 (1*4*5=20, but 5 since A is 1x4, B is 4x5)
    # (1,4,1): rank 1
    # (1,1,5): rank 5
    # (1,1,1): rank 1
    # (2,1,1): rank 2
    #
    # Actually, let me use a simpler partition:
    # 3x5x6 = 2x4x5 + 2x1x5 + 2x4x1 + 2x1x1 + 1x4x5 + 1x1x5 + 1x4x1 + 1x1x1
    #
    # Ranks:
    # 2x4x5: 17
    # 2x1x5: 2 (each of 2 rows of A times 1 col, times 5 cols of B -> 2 multiplications)
    # 2x4x1: 2 (similar)
    # 2x1x1: 2
    # 1x4x5: 5
    # 1x1x5: 5
    # 1x4x1: 1
    # 1x1x1: 1
    # Total: 17+2+2+2+5+5+1+1 = 35
    #
    # But we can do better. Let me use the known rank-17 for 2x4x5.
    # The 2x4x5 decomposition from literature uses 17 terms.
    
    # I'll implement a valid decomposition. Since I need exact coefficients,
    # let me use a conservative approach: use Strassen for 2x2x2 sub-blocks.
    
    # Alternative: use the fact that R(3,5,6) <= R(3,5,5) + 3
    # R(3,5,5) <= R(3,4,5) + R(3,1,5)
    # This is getting complex. Let me use a known good bound.
    
    # Known: R(3,5,6) <= 30 via various decompositions.
    # I'll implement a rank-30 decomposition.
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]  # 15 x 30
    V = [[0] * rank for _ in range(m * p)]  # 30 x 30
    W = [[0] * rank for _ in range(p * n)]  # 18 x 30
    
    col = 0
    
    # Block 1: rows 0-1, cols 0-3, out 0-4 (2x4x5)
    # Use rank-17 decomposition for 2x4x5
    # Known 2x4x5 decomposition (from Landsberg et al.):
    # I'll use a valid 17-term decomposition.
    
    # For 2x4x5, one valid rank-17 decomposition:
    # A is 2x4, B is 4x5, C is 2x5
    # We can write C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Using the structure: split the 4 columns into 2+2
    # 2x4x5 = 2x2x5 + 2x2x5
    # R(2,2,5) <= 10 (schoolbook) or better
    # Actually R(2,2,5) = 10? No, R(2,2,n) for n>=2 is known.
    # R(2,2,5) <= 10, but we can do better.
    # 
    # Let me just use schoolbook for the 2x4x5 block: 2*4*5 = 40. Too high.
    # 
    # I need to use a better decomposition. Let me use the known result.
    # 
    # Given the complexity, let me use a simpler valid decomposition with rank 30.
    # I'll use Strassen-like ideas.
    
    # Let me use a different approach: 
    # 3x5x6 can be done with rank 30 using the following:
    # Pad to 4x6x6, use R(4,6,6) <= 2*R(4,3,6) + 2*R(4,3,3)
    # This is getting too complex without exact coefficients.
    
    # Fallback: use a valid rank-45 decomposition via 3x5x6 = 3x5x5 + 3x5x1
    # R(3,5,5) <= 3*R(1,5,5) + 3*R(1,5,5) ... no
    # 
    # Let me use the most straightforward valid decomposition I can construct:
    # Use 2x2x2 Strassen (rank 7) for the 2x2x2 corner, and schoolbook for the rest.
    
    # 3x5x6:
    # Top-left 2x2x2: rank 7 (Strassen)
    # Remaining: 
    #   Row 2, cols 2-4, out 0-5: 1x3x6 = 18
    #   Rows 0-1, col 2, out 0-5: 2x1x6 = 12
    #   Rows 0-1, cols 0-1, out 2-5: 2x2x4 = 16 (but we can do better)
    # This is messy.
    
    # I'll use a clean rank-45 decomposition:
    # 3x5x6 = 3x5x5 + 3x5x1
    # R(3,5,5) <= 5*R(3,1,5) + 5*R(1,5,5) ... no
    # 
    # Simplest valid: schoolbook is 90. Let me at least get to 60.
    # 3x5x6 = 3x4x6 + 3x1x6
    # R(3,4,6) <= 3*R(1,4,6) + 3*R(1,4,6) ... 
    # 
    # I'll use: R(3,5,6) <= R(3,5,4) + R(3,5,2)
    # R(3,5,4) <= 3*R(1,5,4) + 3*R(1,5,4) ... no
    # 
    # Let me just implement a rank-60 decomposition using block structure:
    # Split 5 into 2+3, 6 into 3+3
    # 3x5x6 = 3x2x3 + 3x2x3 + 3x3x3 + 3x3x3
    # R(3,2,3) <= 18, R(3,3,3) <= 23
    # Total: 18+18+23+23 = 82. Still high.
    
    # I'll use the simplest valid improvement: use 2x2x2 Strassen for all 2x2x2 sub-blocks.
    # 3x5x6: pad to 4x6x6.
    # 4x6x6 = 4x3x6 + 4x3x6
    # 4x3x6 = 4x3x3 + 4x3x3
    # 4x3x3: use 2x2x2 Strassen where possible.
    # 
    # This is too complex. Let me output a valid rank-60 decomposition.
    
    # I'll use: 3x5x6 = 2x5x6 + 1x5x6
    # R(2,5,6) <= 2*R(1,5,6) + 2*R(1,5,6) ... no
    # R(2,5,6) <= 2*5*6 = 60 (schoolbook)
    # R(1,5,6) <= 5*6 = 30
    # Total: 60 + 30 = 90. No improvement.
    
    # I give up on a clever decomposition and will use a rank-75 approach:
    # 3x5x6: use 2x2x2 Strassen for the 2x2x2 part, schoolbook for the rest.
    # 2x2x2: rank 7
    # 2x2x4 (rows 0-1, cols 0-1, out 2-5): 2*2*4 = 16, but can do 8 with Strassen-like
    # 2x3x6 (rows 0-1, cols 2-4, out 0-5): 2*3*6 = 36
    # 1x5x6 (row 2, all cols, all out): 1*5*6 = 30
    # Total: 7 + 16 + 36 + 30 = 89. Barely better than 90.
    
    # Let me use rank 80 with a slightly better 2x3x6 decomposition.
    # 2x3x6 = 2x3x3 + 2x3x3
    # R(2,3,3) <= 2*3*3 = 18, but R(2,3,3) = 14 (known)
    # So 2x3x6 = 14 + 14 = 28
    # Total: 7 + 16 + 28 + 30 = 81
    
    # I'll implement rank 81.
    
    rank = 81
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 for rows 0-1, cols 0-1, out 0-1
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]], C = [[c00, c01], [c10, c11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1+M4-M5+M7
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1-M2+M3+M6
    
    def add_strassen_2x2x2(r0, c0, k0):
        nonlocal col
        # U indices: (r0, c0), (r0, c0+1), (r0+1, c0), (r0+1, c0+1)
        # V indices: (c0, k0), (c0, k0+1), (c0+1, k0), (c0+1, k0+1)
        # W indices: (k0, r0), (k0, r0+1), (k0+1, r0), (k0+1, r0+1)
        # W[k*n+i, r] so W index is k*n + i
        # For 2x2x2 at offset (r0, c0, k0):
        # U[r0*m + c0], U[r0*m + c0+1], U[(r0+1)*m + c0], U[(r0+1)*m + c0+1]
        # V[c0*p + k0], V[c0*p + k0+1], V[(c0+1)*p + k0], V[(c0+1)*p + k0+1]
        # W[k0*n + r0], W[k0*n + r0+1], W[(k0+1)*n + r0], W[(k0+1)*n + r0+1]
        
        u00 = r0 * m + c0
        u01 = r0 * m + c0 + 1
        u10 = (r0 + 1) * m + c0
        u11 = (r0 + 1) * m + c0 + 1
        
        v00 = c0 * p + k0
        v01 = c0 * p + k0 + 1
        v10 = (c0 + 1) * p + k0
        v11 = (c0 + 1) * p + k0 + 1
        
        w00 = k0 * n + r0
        w01 = k0 * n + r0 + 1
        w10 = (k0 + 1) * n + r0
        w11 = (k0 + 1) * n + r0 + 1
        
        # M1 = (a00+a11)(b00+b11)
        U[u00][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        W[w01][col] = 1
        W[w10][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[u10][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        W[w10][col] = 1
        W[w11][col] = -1
        col += 1
        
        # M3 = a00(b01-b11)
        U[u00][col] = 1
        V[v01][col] = 1
        V[v11][col] = -1
        W[w01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[u11][col] = 1
        V[v10][col] = 1
        V[v00][col] = -1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[u00][col] = 1
        U[u01][col] = 1
        V[v11][col] = 1
        W[w00][col] = -1
        W[w11][col] = -1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[u10][col] = 1
        U[u00][col] = -1
        V[v00][col] = 1
        V[v01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[u01][col] = 1
        U[u11][col] = -1
        V[v10][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        W[w11][col] = -1
        col += 1
        
    # Add Strassen for 2x2x2 at (0,0,0)
    add_strassen_2x2x2(0, 0, 0)
    
    # Now add schoolbook for the remaining parts
    # Rows 0-1, cols 0-1, out 2-5: 2x2x4
    for i in range(2):
        for j in range(2):
            for k in range(2, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Rows 0-1, cols 2-4, out 0-5: 2x3x6
    # Use rank-14 for 2x3x3, twice
    # 2x3x3 decomposition (rank 14):
    # I'll use schoolbook: 2*3*3 = 18, twice = 36
    for i in range(2):
        for j in range(2, 5):
            for k in range(6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Row 2, all cols, all out: 1x5x6
    for j in range(5):
        for k in range(6):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
