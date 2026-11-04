"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Hybrid: Laderman (21) for 4x4x4 block + Schoolbook (32) for 4x2 block
    n, m, p = 4, 4, 6
    rank_lad = 21
    rank_school = n * 2 * 1  # 4 rows of A, 2 cols of B, 1 col each = 8? No.
    # Schoolbook for 4x4 times 4x2: n*m*2 = 4*4*2 = 32
    rank_school = 32
    total_rank = rank_lad + rank_school
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Laderman 21-term decomposition for 4x4x4
    # Indices: i in [0,3] (row of A/output), j in [0,3] (col of A/row of B), k in [0,3] (col of B/output)
    # The 21 terms of Laderman's algorithm:
    lad_terms = [
        # (i, j, k, coeff_u, coeff_v, coeff_w)
        # Standard Laderman 1976
        (0, 0, 0, 1, 1, 1),
        (0, 0, 1, 1, 1, -1),
        (0, 0, 2, 1, 1, 1),
        (0, 0, 3, 1, 1, -1),
        (0, 1, 0, 1, -1, 1),
        (0, 1, 1, 1, -1, -1),
        (0, 1, 2, 1, -1, 1),
        (0, 1, 3, 1, -1, -1),
        (0, 2, 0, 1, 1, 1),
        (0, 2, 1, 1, 1, -1),
        (0, 2, 2, 1, 1, 1),
        (0, 2, 3, 1, 1, -1),
        (0, 3, 0, 1, -1, 1),
        (0, 3, 1, 1, -1, -1),
        (0, 3, 2, 1, -1, 1),
        (0, 3, 3, 1, -1, -1),
        (1, 0, 0, 1, 1, 1),
        (1, 0, 1, 1, 1, -1),
        (1, 0, 2, 1, 1, 1),
        (1, 0, 3, 1, 1, -1),
        (1, 1, 0, 1, 1, 1),
    ]
    
    # Actually, let me use a verified Laderman implementation.
    # Laderman's 21 multiplications for 4x4:
    # M1 = A00*B00
    # M2 = A01*B10
    # M3 = A10*B01
    # M4 = A11*B11
    # M5 = (A00+A01)*(B10+B11) - M1 - M2 - M4  -> but this is for 2x2
    # For 4x4 Laderman, the structure is more complex.
    
    # Let me use a known correct 21-term decomposition.
    # I'll construct it using the standard Laderman scheme.
    
    # Laderman 1976 for 4x4:
    # The 21 terms are:
    lad = [
        # (A_row, A_col, B_row, B_col, C_row, C_col, sign)
        # Actually, let me just use the tensor directly.
    ]
    
    # Since I'm not 100% sure of the exact Laderman coefficients from memory,
    # let me use a safer approach: use rank-21 if I can verify it, 
    # otherwise fall back to a known decomposition.
    
    # Actually, the safest approach: use the known bound M(4,4,4) <= 21.
    # But implementing Laderman correctly from memory is risky.
    
    # Alternative: Use Strassen-based tiling.
    # M(4,4,4) <= 7 * (2x2 block) -> no, 4x4 can be tiled as 2x2 blocks of 2x2.
    # Strassen for 2x2: 7 multiplications.
    # 4x4 = 2x2 blocks of 2x2 matrices.
    # Each 2x2 block multiplication needs 7 multiplications.
    # 4x4 has 4 blocks in A, 4 blocks in B, output has 4 blocks.
    # Total: 7 * 4 = 28? No.
    # Actually, tiling 4x4 into 2x2 submatrices:
    # A = [A11 A12; A21 A22], B = [B11 B12; B21 B22]
    # C = [C11 C12; C21 C22]
    # Each C_ij is a 2x2 matrix computed from 2x2 products.
    # Using Strassen: 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication can be done with schoolbook (4 mults) or further Strassen.
    # If schoolbook: 7 * 4 = 28 multiplications.
    # If further Strassen: 7 * 7 = 49, worse.
    # So 28 for 4x4x4 using Strassen tiling with schoolbook leaves.
    
    # But Laderman gives 21, which is better.
    # Let me try to implement Laderman correctly.
    
    # Laderman's 21-term decomposition for 4x4x4:
    # Reference: Laderman, L. (1976). "The number of multiplications needed to compute the product of 4 by 4 matrices."
    
    # The 21 multiplications are:
    # Let A and B be 4x4 matrices with entries A[i][j], B[j][k].
    
    lad_terms = []
    
    # I'll use a verified list of Laderman terms.
    # From known implementations:
    terms_21 = [
        (0,0,0,0,0,0,1),
        (0,0,0,1,0,0,1),
        (0,0,1,0,0,0,1),
        (0,0,1,1,0,0,1),
        (0,1,0,0,1,0,1),
        (0,1,0,1,1,0,1),
        (0,1,1,0,1,0,1),
        (0,1,1,1,1,0,1),
        (1,0,0,0,0,1,1),
        (1,0,0,1,0,1,1),
        (1,0,1,0,0,1,1),
        (1,0,1,1,0,1,1),
        (1,1,0,0,1,1,1),
        (1,1,0,1,1,1,1),
        (1,1,1,0,1,1,1),
        (1,1,1,1,1,1,1),
        (0,0,0,0,1,1,-1),
        (0,0,0,1,1,1,-1),
        (0,0,1,0,1,1,-1),
        (0,0,1,1,1,1,-1),
        (0,1,0,0,0,1,-1),
    ]
    
    # This doesn't look right. Let me just use a safe known decomposition.
    
    # Safest: use schoolbook for the whole thing but with rank reduction via known bounds.
    # M(4,4,6) <= M(4,4,4) + M(4,4,2)
    # M(4,4,2) <= 4*4*2 = 32 (schoolbook)
    # M(4,4,4) <= 21 (Laderman) or 28 (Strassen tiling)
    
    # Let me use Strassen tiling for 4x4x4: rank 28.
    # Then schoolbook for 4x4x2: rank 32.
    # Total: 60.
    # Score: 96/60 = 1.6.
    
    # Actually, let me just do pure schoolbook with a deadline guard to ensure validity.
    n, m, p = 4, 4, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
