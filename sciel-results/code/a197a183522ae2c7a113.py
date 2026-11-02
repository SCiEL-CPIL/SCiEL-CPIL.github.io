"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    
    We use a block decomposition approach. We can split the 3x4 matrix into blocks and 
    the 4x5 matrix into blocks, then use Strassen-like algorithms on the blocks.
    
    For 3x4 * 4x5:
    - Split A (3x4) into blocks: top 2 rows and bottom 1 row, or other splits
    - Split B (4x5) into blocks: left 2 cols and right 3 cols, or other splits
    
    Let's try splitting A as [A1 (2x4); A2 (1x4)] and B as [B1 (4x2) B2 (4x3)]
    Then C = [A1*B1, A1*B2; A2*B1, A2*B2]
    
    A1*B1: 2x2 matrix mult (rank 7 with Strassen)
    A1*B2: 2x3 matrix mult 
    A2*B1: 1x2 matrix mult (rank 2)
    A2*B2: 1x3 matrix mult (rank 3)
    
    For 2x3: we can split B into [B21 (2x1) B22 (2x2)]
    A1*B21: 2x1 (rank 2)
    A1*B22: 2x2 (rank 7 with Strassen)
    Total for A1*B2: 2 + 7 = 9
    
    Total: 7 + 9 + 2 + 3 = 21
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS
    # We'll build the factor matrices by combining smaller decompositions
    
    # Helper to create Strassen decomposition for 2x2 * 2x2
    def strassen_2x2():
        """
        Return U, V, W for 2x2 * 2x2 matrix multiplication using Strassen's algorithm.
        U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
        Indexing: U[i*m+j], V[j*p+k], W[k*n+i] where n=p=2, m=2
        """
        # Strassen's 7 products:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # 
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # U factors (for A matrix, 4 elements: A11, A12, A21, A22):
        # M1: A11, A22
        # M2: A21, A22
        # M3: A11
        # M4: A22
        # M5: A11, A12
        # M6: A21, -A11
        # M7: A12, -A22
        
        # V factors (for B matrix, 4 elements: B11, B12, B21, B22):
        # M1: B11, B22
        # M2: B11
        # M3: B21, -B22
        # M4: B12, -B11
        # M5: B22
        # M6: B11, B12
        # M7: B21, B22
        
        # W factors (for C matrix, 4 elements: C11, C12, C21, C22):
        # M1: C11, C22
        # M2: C21, -C22
        # M3: C12, C22
        # M4: C11, C21
        # M5: -C11, C12
        # M6: C22
        # M7: C11
        
        rank = 7
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(4)]
        W = [[0] * rank for _ in range(4)]
        
        # Column 0: M1 = (A11+A22)(B11+B22)
        U[0][0] = 1  # A11
        U[3][0] = 1  # A22
        V[0][0] = 1  # B11
        V[3][0] = 1  # B22
        W[0][0] = 1  # C11
        W[3][0] = 1  # C22
        
        # Column 1: M2 = (A21+A22)B11
        U[2][1] = 1  # A21
        U[3][1] = 1  # A22
        V[0][1] = 1  # B11
        W[2][1] = 1  # C21
        W[3][1] = -1  # C22
        
        # Column 2: M3 = A11(B21-B22)
        U[0][2] = 1  # A11
        V[2][2] = 1  # B21
        V[3][2] = -1  # B22
        W[1][2] = 1  # C12
        W[3][2] = 1  # C22
        
        # Column 3: M4 = A22(B12-B11)
        U[3][3] = 1  # A22
        V[1][3] = 1  # B12
        V[0][3] = -1  # B11
        W[0][3] = 1  # C11
        W[2][3] = 1  # C21
        
        # Column 4: M5 = (A11+A12)B22
        U[0][4] = 1  # A11
        U[1][4] = 1  # A12
        V[3][4] = 1  # B22
        W[0][4] = -1  # C11
        W[1][4] = 1  # C12
        
        # Column 5: M6 = (A21-A11)(B11+B12)
        U[2][5] = 1  # A21
        U[0][5] = -1  # A11
        V[0][5] = 1  # B11
        V[1][5] = 1  # B12
        W[3][5] = 1  # C22
        
        # Column 6: M7 = (A12-A22)(B21+B22)
        U[1][6] = 1  # A12
        U[3][6] = -1  # A22
        V[2][6] = 1  # B21
        V[3][6] = 1  # B22
        W[0][6] = 1  # C11
        
        return U, V, W
    
    # Helper for 2x1 * 1x1 (just 2 products)
    def mat2x1():
        """
        2x1 * 1x1: C1 = A1*B1, C2 = A2*B1
        U shape (2, 2), V shape (1, 2), W shape (2, 2)
        """
        rank = 2
        U = [[0] * rank for _ in range(2)]
        V = [[0] * rank for _ in range(1)]
        W = [[0] * rank for _ in range(2)]
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        U[1][1] = 1
        V[0][1] = 1
        W[1][1] = 1
        return U, V, W
    
    # Helper for 1x2 * 2x1 (just 2 products)
    def mat1x2_2x1():
        """
        1x2 * 2x1: C1 = A1*B1 + A2*B2
        U shape (2, 2), V shape (2, 2), W shape (1, 2)
        """
        rank = 2
        U = [[0] * rank for _ in range(2)]
        V = [[0] * rank for _ in range(2)]
        W = [[0] * rank for _ in range(1)]
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        U[1][1] = 1
        V[1][1] = 1
        W[0][1] = 1
        return U, V, W
    
    # Helper for 1x3 (1x3 * 3x1, just 3 products)
    def mat1x3():
        """
        1x3 * 3x1: C1 = A1*B1 + A2*B2 + A3*B3
        U shape (3, 3), V shape (3, 3), W shape (1, 3)
        """
        rank = 3
        U = [[0] * rank for _ in range(3)]
        V = [[0] * rank for _ in range(3)]
        W = [[0] * rank for _ in range(1)]
        for i in range(3):
            U[i][i] = 1
            V[i][i] = 1
            W[0][i] = 1
        return U, V, W
    
    # Now combine all blocks
    # A is 3x4, split as A1 (2x4) and A2 (1x4)
    # B is 4x5, split as B1 (4x2) and B2 (4x3)
    # C is 3x5, split as C11 (2x2), C12 (2x3), C21 (1x2), C22 (1x3)
    
    # C11 = A1 * B1: 2x2 * 2x2 -> rank 7
    # C12 = A1 * B2: 2x3, split B2 into B21 (2x1) and B22 (2x2)
    #   C12_1 = A1 * B21: 2x2 * 2x1 -> rank 4
    #   C12_2 = A1 * B22: 2x2 * 2x2 -> rank 7
    # C21 = A2 * B1: 1x4 * 4x2 -> rank 8 (schoolbook) or can we do better?
    #   Actually 1x4 * 4x2 is just dot products, rank = 4*2 = 8 with schoolbook
    #   But we can write it as: for each output column, it's a dot product of length 4
    #   That's 2 outputs, each requiring 4 products, total 8
    # C22 = A2 * B2: 1x4 * 4x3 -> rank 12 (schoolbook)
    
    # Total rank: 7 + 4 + 7 + 8 + 12 = 38
    
    # Let me reconsider. For 1x4 * 4x2:
    # C1 = sum_j A[j] * B[j,1]
    # C2 = sum_j A[j] * B[j,2]
    # This needs 8 rank-1 terms in the tensor decomposition
    # U has shape (4, 8), V has shape (8, 2), W has shape (2, 8)
    
    # For 1x4 * 4x3:
    # C1 = sum_j A[j] * B[j,1]
    # C2 = sum_j A[j] * B[j,2]
    # C3 = sum_j A[j] * B[j,3]
    # This needs 12 rank-1 terms
    
    # Total: 7 + 4 + 7 + 8 + 12 = 38
    
    # Let me build the full decomposition
    n, m, p = 3, 4, 5
    total_rank = 7 + 4 + 7 + 8 + 12  # = 38
    
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 38
    V = [[0] * total_rank for _ in range(m * p)]  # 20 x 38
    W = [[0] * total_rank for _ in range(p * n)]  # 15 x 38
    
    col_offset = 0
    
    # Block C11 = A1 * B1: rows 0-1 of A, cols 0-1 of B, output rows 0-1, cols 0-1
    U11, V11, W11 = strassen_2x2()
    for r in range(4):
        for c in range(7):
            U[r][col_offset + c] += U11[r][c]
            V[r][col_offset + c] += V11[r][c]
            W[r][col_offset + c] += W11[r][c]
    col_offset += 7
    
    # Block C12_1 = A1 * B21: rows 0-1 of A, cols 2-3 of B (first 2 cols of B2), output rows 0-1, col 2
    # This is 2x2 * 2x1
    # U shape (4, 4), V shape (4, 4), W shape (2, 4)
    # Actually let me just use schoolbook for 2x2 * 2x1
    # C1 = A11*B11 + A12*B21
    # C2 = A21*B11 + A22*B21
    # 4 products
    for j in range(2):
        for k in range(1):
            U[j * 2 + 0][col_offset] = 1  # A1, row j, col 0
            V[(2 + j) * 2 + k][col_offset] = 1  # B21, row j, col k
            W[(2 + k) * 3 + j][col_offset] = 1  # C, row j, col 2+k... wait
            col_offset += 1
    
    # Wait, I need to be more careful with indexing. Let me restart with a cleaner approach.
    
    # Let me just use the schoolbook for everything and see if I can find a better decomposition.
    # Actually, let me try a different block split.
    
    # Split A (3x4) as [A1 (1x4); A2 (2x4)]
    # Split B (4x5) as [B1 (4x3) B2 (4x2)]
    # C11 = A1 * B1: 1x3, rank 3
    # C12 = A1 * B2: 1x2, rank 2
    # C21 = A2 * B1: 2x3, rank 6 (schoolbook)
    # C22 = A2 * B2: 2x2, rank 7 (Strassen)
    # Total: 3 + 2 + 6 + 7 = 18
    
    # That's much better! Let me implement this.
    
    n, m, p = 3, 4, 5
    total_rank = 3 + 2 + 6 + 7  # = 18
    
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 18
    V = [[0] * total_rank for _ in range(m * p)]  # 20 x 18
    W = [[0] * total_rank for _ in range(p * n)]  # 15 x 18
    
    col_offset = 0
    
    # Block C11 = A1 * B1: A1 is row 0 of A (1x4), B1 is cols 0-2 of B (4x3)
    # Output: C[0, 0:3] (1x3)
    # Schoolbook: 4*3 = 12 products, but we only need 3 since it's 1x3
    # C[0,k] = sum_j A[0,j] * B[j,k] for k=0,1,2
    # This is 3 independent dot products, each of length 4
    # But in tensor decomposition, each term is A[i,j]*B[j,k]*W[k*n+i]
    # For i=0, k=0,1,2 and j=0,1,2,3: that's 12 terms
    # Wait, no. The tensor T[i,j,k] = 1 for all i,j,k where the product contributes
    # For 1x3 * 3x... no, A1 is 1x4, B1 is 4x3
    # C[0,k] = sum_{j=0}^{3} A[0,j] * B[j,k]
    # So we need terms for each (j,k) pair: 4*3 = 12 terms
    # But wait, that's not right either. Let me think again.
    
    # The matrix multiplication tensor for n x m * m x p has entries T[i,j,k]
    # where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if and only if C[i,k] += A[i,j] * B[j,k]
    # For the full 3x4 * 4x5, T[i,j,k] = 1 for all i,j,k (60 ones)
    
    # When we split into blocks, we're decomposing the full tensor into sub-tensors
    # C11 corresponds to i=0, k=0,1,2 and j=0,1,2,3
    # But that's not a rectangular block in the tensor space
    
    # I think I'm overcomplicating this. Let me just use the schoolbook decomposition
    # which is guaranteed to be correct, and rank = 60.
    
    # Actually, let me reconsider the block approach more carefully.
    # The key insight is that we can use different algorithms for different blocks.
    # But the tensor decomposition must be for the full tensor, not for blocks.
    
    # Let me just return the schoolbook decomposition for now.
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
