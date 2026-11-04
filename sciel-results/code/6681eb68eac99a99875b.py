"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We'll compute C = A * B where A is 4x4, B is 4x6, C is 4x6
    # Split B into B1 (4x4) and B2 (4x2)
    # C = A * B1 + A * B2
    # A * B1: use Strassen recursively, rank 49
    # A * B2: schoolbook, rank 32
    # Total rank: 81
    
    rank = 81
    U = [[0] * rank for _ in range(n * m)]  # 16 x 81
    V = [[0] * rank for _ in range(m * p)]  # 24 x 81
    W = [[0] * rank for _ in range(p * n)]  # 24 x 81
    
    col = 0
    
    # Helper: add a Strassen 2x2 x 2x2 decomposition at given offsets
    # A_base: (i, j) of top-left of A in A
    # B_base: (i, j) of top-left of B in B
    # C_base: (i, j) of top-left of C in C
    # All blocks are 2x2
    def add_strassen_2x2(A_base, B_base, C_base):
        nonlocal col
        iA, jA = A_base
        iB, jB = B_base
        iC, jC = C_base
        
        # Strassen's 7 products for 2x2 x 2x2
        # M1 = (a+d)(e+h)
        U[iA * m + jA][col] = 1
        U[(iA + 1) * m + (jA + 1)][col] = 1
        V[iB * p + jB][col] = 1
        V[(iB + 1) * p + (jB + 1)][col] = 1
        W[jC * n + iC][col] = 1
        W[(jC + 1) * n + (iC + 1)][col] = 1
        col += 1
        
        # M2 = (c+d)e
        U[(iA + 1) * m + jA][col] = 1
        U[(iA + 1) * m + (jA + 1)][col] = 1
        V[iB * p + jB][col] = 1
        W[jC * n + (iC + 1)][col] = 1
        W[(jC + 1) * n + (iC + 1)][col] = 1
        col += 1
        
        # M3 = a(f-h)
        U[iA * m + jA][col] = 1
        V[iB * p + (jB + 1)][col] = 1
        V[(iB + 1) * p + (jB + 1)][col] = -1
        W[(jC + 1) * n + iC][col] = 1
        W[(jC + 1) * n + (iC + 1)][col] = -1
        col += 1
        
        # M4 = d(g-e)
        U[(iA + 1) * m + (jA + 1)][col] = 1
        V[(iB + 1) * p + jB][col] = 1
        V[iB * p + jB][col] = -1
        W[jC * n + iC][col] = -1
        W[jC * n + (iC + 1)][col] = 1
        col += 1
        
        # M5 = (a+b)h
        U[iA * m + jA][col] = 1
        U[iA * m + (jA + 1)][col] = 1
        V[(iB + 1) * p + (jB + 1)][col] = 1
        W[jC * n + iC][col] = 1
        W[(jC + 1) * n + iC][col] = 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[(iA + 1) * m + jA][col] = 1
        U[iA * m + jA][col] = -1
        V[iB * p + jB][col] = 1
        V[iB * p + (jB + 1)][col] = 1
        W[(jC + 1) * n + (iC + 1)][col] = -1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[iA * m + (jA + 1)][col] = 1
        U[(iA + 1) * m + (jA + 1)][col] = -1
        V[(iB + 1) * p + jB][col] = 1
        V[(iB + 1) * p + (jB + 1)][col] = 1
        W[jC * n + iC][col] = -1
        col += 1
        
        return col
    
    # For 4x4 x 4x4, use Strassen with 2x2 blocks
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # Each block is 2x2
    
    # We need to compute 7 block products, each of which is a 2x2 x 2x2 multiplication
    # But the "A" and "B" matrices for each block product are linear combinations of the original blocks
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M6
    
    # For each block product, we need to decompose it into 7 scalar products using Strassen
    # But the "A" matrix is a sum/difference of 2x2 blocks, so we need to handle this carefully
    
    # Let's define a helper that takes a list of (coeff, block_pos) pairs for A and B
    # and adds the Strassen decomposition with those coefficients
    
    def add_strassen_with_coeffs(A_coeffs, B_coeffs, C_coeffs, C_pos):
        """
        A_coeffs: list of (coeff, (i,j)) where (i,j) is top-left of 2x2 block
        B_coeffs: list of (coeff, (i,j)) where (i,j) is top-left of 2x2 block
        C_coeffs: list of (coeff, (i,j)) where (i,j) is top-left of 2x2 block
        C_pos: (i,j) top-left of result C block
        """
        iC, jC = C_pos
        nonlocal col
        
        # For each of the 7 Strassen products, we need to add contributions from all A and B coefficients
        # This is complex, so let's use a different approach: directly enumerate the 49 terms
        
        # Actually, let's just use the fact that Strassen on 2x2 blocks gives rank 7 per block product
        # and we have 7 block products, so total rank 49 for the 4x4 x 4x4 part
        
        # Let me implement this more carefully by directly constructing the 49 terms
        
        pass
    
    # Given the complexity, let me use a simpler approach:
    # For the 4x4 x 4x4 part, I'll use the known rank-49 decomposition
    # For the 4x4 x 4x2 part, I'll use schoolbook with rank 32
    
    # Total rank: 49 + 32 = 81
    
    # Let me implement the 4x4 x 4x4 Strassen decomposition directly
    
    # The 7 Strassen products at the block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mi is a 2x2 x 2x2 product. For each Mi, I need to decompose it into 7 scalar products.
    # The "A" matrix for Mi is a linear combination of A blocks, and similarly for "B".
    
    # Let me define the 7 block products with their A and B coefficient patterns:
    
    # M1: A = A00 + A11, B = B00 + B11
    # A coeffs: [(1, (0,0)), (1, (2,2))]
    # B coeffs: [(1, (0,0)), (1, (2,2))]
    
    # M2: A = A10 + A11, B = B00
    # A coeffs: [(1, (2,0)), (1, (2,2))]
    # B coeffs: [(1, (0,0))]
    
    # M3: A = A00, B = B01 - B11
    # A coeffs: [(1, (0,0))]
    # B coeffs: [(1, (0,2)), (-1, (2,2))]
    
    # M4: A = A11, B = B10 - B00
    # A coeffs: [(1, (2,2))]
    # B coeffs: [(1, (2,0)), (-1, (0,0))]
    
    # M5: A = A00 + A01, B = B11
    # A coeffs: [(1, (0,0)), (1, (0,2))]
    # B coeffs: [(1, (2,2))]
    
    # M6: A = A10 - A00, B = B00 + B01
    # A coeffs: [(1, (2,0)), (-1, (0,0))]
    # B coeffs: [(1, (0,0)), (1, (0,2))]
    
    # M7: A = A01 - A11, B = B10 + B11
    # A coeffs: [(1, (0,2)), (-1, (2,2))]
    # B coeffs: [(1, (2,0)), (1, (2,2))]
    
    # For each block product, I'll add 7 Strassen terms with appropriate coefficients
    
    # This is still complex. Let me just use schoolbook for the entire thing to ensure correctness.
    
    rank = n * m * p  # 96
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
