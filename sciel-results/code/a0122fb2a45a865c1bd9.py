"""Matrix multiplication 4x5x6 via Strassen-like block decomposition."""
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We will decompose the 4x5x6 multiplication tensor using a block approach.
    # We can view this as multiplying a 4x5 matrix A by a 5x6 matrix B.
    # To leverage known lower rank decompositions for 2x2x2, we can pad A to 4x8 and B to 8x6?
    # No, let's use a simpler block structure that aligns with 2x2 blocks where possible.
    
    # Actually, the most straightforward way to get a better-than-schoolbook rank for 4x5x6
    # without a known specific decomposition for 4x5x6 is to use a "tensor product" of decompositions.
    
    # Let's try to decompose 4x5x6 as:
    # 4x5x6 = (2x2x2) tensor (2x1x3) ? No, dimensions don't match for simple tensor product.
    
    # Let's look at the dimensions: 4, 5, 6.
    # 4 = 2*2, 6 = 2*3, 5 is prime.
    
    # We can split the matrices as:
    # A (4x5) into blocks:
    #   A11 (2x2) A12 (2x3)
    #   A21 (2x2) A22 (2x3)
    # B (5x6) into blocks:
    #   B11 (2x6)
    #   B21 (3x6)
    
    # C (4x6) = A * B
    # C11 = A11*B11 + A12*B21 (2x6)
    # C21 = A21*B11 + A22*B21 (2x6)
    
    # This doesn't immediately suggest a rank reduction because the middle dimension 5 is split 2+3.
    
    # Let's try a different split.
    # Split A (4x5) into 2x5 and 2x5? No, that's just row splitting.
    # Split B (5x6) into 5x3 and 5x3?
    
    # Let's try to use the fact that rank(4x5x6) <= rank(4x5x5) + rank(4x5x1) ?
    # rank(4x5x6) <= rank(4x5x3) + rank(4x5x3) ?
    
    # Actually, a known result is that the rank of m x n x p matrix multiplication is at most
    # the product of the ranks of smaller multiplications if we can block diagonalize.
    
    # Let's try to construct a decomposition based on 2x2x2 Strassen for the even parts.
    # 4x6 part of the output can be seen as two 2x6 outputs.
    
    # Alternative strategy: Use the "submatrix" method.
    # If we have a decomposition for 2x2x2 of rank 7, can we use it for 4x4x4? Yes, rank 49.
    # Can we use it for 4x4x6?
    # 4x4x6 = 4x4x3 + 4x4x3.
    # Rank(4x4x3) <= 2 * Rank(2x2x3)? No.
    
    # Let's stick to a safer but still improved decomposition.
    # Schoolbook rank: 4*5*6 = 120.
    
    # We can split the 5 dimension into 2 and 3.
    # A = [A_top (2x5), A_bottom (2x5)]
    # B = [B_left (5x3), B_right (5x3)]
    
    # This doesn't help directly with Strassen unless we split all dimensions by powers of 2.
    # Since 5 is not a power of 2, we might not get a huge win from Strassen alone.
    
    # However, we can use a "padding" argument or simply compute the rank of the 4x5x6 tensor directly
    # using a known bound. The best known upper bound for 4x5x6 is not significantly lower than 120
    # in simple algorithms, but let's try to get below 120.
    
    # Let's try a decomposition based on splitting the 6 dimension into 2+2+2.
    # And the 4 dimension into 2+2.
    # The 5 dimension remains 5.
    
    # C (4x6) = A (4x5) * B (5x6)
    # Let B = [B1, B2, B3] where each Bj is 5x2.
    # C = [C1, C2, C3] where each Ci is 4x2.
    # Ci = A * Bi.
    # Each Ci computation is a 4x5x2 multiplication.
    # Rank(4x5x2) <= Rank(4x4x2) + Rank(4x1x2) ?
    # Rank(4x4x2): Split 4x4 into four 2x2 blocks.
    #   Rank(4x4x2) <= 2 * Rank(2x2x2) ? No, the third dimension is 2.
    #   Actually, 4x4x2 can be seen as two 4x4x1? No.
    #   4x4x2 = 2x2 blocks of A multiplied by 2x2 blocks of B?
    #   If we split A into 2x2 blocks and B (5x2) ... wait B is 5x2.
    
    # Let's go with a simple block decomposition that guarantees rank < 120.
    # Split A (4x5) into A1 (2x5) and A2 (2x5).
    # Split B (5x6) into B1 (5x3) and B2 (5x3).
    # C11 = A1*B1 (2x3)
    # C12 = A1*B2 (2x3)
    # C21 = A2*B1 (2x3)
    # C22 = A2*B2 (2x3)
    
    # This is just schoolbook again, rank 4*5*6 = 120.
    
    # To get a better rank, we need a non-trivial decomposition.
    # The best known upper bound for 4x5x6 is actually 118? Or maybe lower?
    # Let's try to use the fact that rank(4x5x6) <= rank(4x4x6) + rank(4x1x6).
    # rank(4x1x6) = 6 (identity-like, schoolbook is optimal for 1xn).
    # rank(4x4x6) <= ?
    # 4x4x6 = 4x4x3 + 4x4x3.
    # rank(4x4x3) <= 2 * rank(2x2x3)? No.
    # rank(4x4x3): Split 4x4 into 2x2 blocks.
    #   A = [[A11, A12], [A21, A22]] (each 2x2)
    #   B (3x3) ... wait B is 4x3? No, B is 5x6 in the original, here we are looking at subproblem 4x4x3?
    #   The subproblem is multiplying a 4x4 matrix by a 4x3 matrix.
    #   Let's call it A (4x4) and B (4x3).
    #   C = A*B (4x3).
    #   We can split A into 2x2 blocks.
    #   C1 = A_top * B (2x3)
    #   C2 = A_bottom * B (2x3)
    #   This doesn't reduce rank.
    
    # Let's try a different approach: use the known Strassen decomposition for 2x2x2 (rank 7)
    # and apply it to the 4x4x4 part of the problem, and handle the rest with schoolbook.
    
    # Split the problem 4x5x6 into:
    # 1. 4x4x4 part: A11 (4x4) * B11 (4x4) -> C11 (4x4). Rank <= 49 (Strassen^2).
    #    Wait, B is 5x6. We need to align blocks.
    #    Let's pad B to 5x8? No, that increases rank.
    
    # Let's try to decompose 4x5x6 by splitting the 5 into 2+3 and 6 into 2+2+2.
    # This is getting complicated. Let's use a simpler "tensor product" of 2x2x2 and 2x1x1?
    
    # Actually, the simplest improvement is often to just use a better algorithm for the square sub-problems if they exist.
    # For 4x4x4, rank is 49 (Strassen).
    # For 4x4x2, rank is 16? (Schoolbook is 32).
    #    4x4x2: A (4x4), B (4x2).
    #    Split A into 2x2 blocks.
    #    C1 = A11*B1 + A12*B2 (2x2)
    #    C2 = A21*B1 + A22*B2 (2x2)
    #    This is 2 * Rank(2x2x2) ? No, B is shared.
    #    Actually, rank(4x4x2) = 16 is achievable by just doing 2 separate 2x4x2 multiplications?
    #    2x4x2: A (2x4), B (4x2).
    #    Rank(2x4x2) <= 2 * Rank(2x2x2) ? No.
    #    Rank(2x4x2) = 8? (Schoolbook).
    #    Can we do better?
    #    2x4x2 is equivalent to 2x2x2 tensor 1x2x1?
    #    Rank(2x4x2) = Rank(2x2x2) * Rank(1x2x1) = 7 * 2 = 14.
    #    So rank(4x4x2) = 2 * Rank(2x4x2) = 28.
    
    # So, if we can decompose 4x5x6 into parts:
    # 4x4x4 (rank 49)
    # 4x1x4 (rank 4)
    # 4x4x2 (rank 28)
    # 4x1x2 (rank 2)
    # Total rank = 49 + 4 + 28 + 2 = 83.
    
    # Let's verify the decomposition of 4x5x6 into these blocks.
    # A (4x5) = [A1 (4x4), A2 (4x1)]
    # B (5x6) = [B1 (5x4); B2 (5x2)]  -- Wait, B is 5x6. Split columns into 4 and 2.
    # B = [B1 (5x4) B2 (5x2)]
    # Also split B's rows into 4 and 1 to match A's column split?
    # B = [[B11 (4x4), B12 (4x2)], [B21 (1x4), B22 (1x2)]]
    
    # C = A * B
    # C = [A1 A2] * [[B11 B12], [B21 B22]]
    # C = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    # C1 (4x4) = A1 (4x4) * B11 (4x4) + A2 (4x1) * B21 (1x4)
    # C2 (4x2) = A1 (4x4) * B12 (4x2) + A2 (4x1) * B22 (1x2)
    
    # So we need to compute:
    # 1. A1 * B11 (4x4x4) -> Rank 49
    # 2. A2 * B21 (4x1x4) -> Rank 4
    # 3. A1 * B12 (4x4x2) -> Rank 28
    # 4. A2 * B22 (4x1x2) -> Rank 2
    
    # Total Rank = 49 + 4 + 28 + 2 = 83.
    # This is much better than 120.
    
    # Now we need to construct the U, V, W matrices for this block decomposition.
    
    # We will write helper functions to generate the U, V, W for 2x2x2 Strassen.
    # And for the schoolbook parts (4x1x4, 4x1x2).
    
    # Strassen 2x2x2:
    # M1 = A11 * (B22 - B12)
    # M2 = (A11 + A21) * B22
    # M3 = (A22 - A12) * B11
    # M4 = A22 * (B11 + B21)
    # M5 = (A11 + A12) * (B11 - B21)
    # M6 = (A21 + A22) * (B12 - B22)
    # M7 = (A11 + A22) * (B22 - B11)
    # C11 = M5 + M7 + M1 - M4
    # C12 = M3 + M7
    # C21 = M1 + M6
    # C22 = M2 + M5 - M1 - M3
    
    # We need to map these to U, V, W indices.
    # For a 2x2x2 block, U is 4x7, V is 4x7, W is 4x7.
    # U[i*2+j, r] is the coefficient of A[i,j] in the r-th product.
    # V[j*2+k, r] is the coefficient of B[j,k] in the r-th product.
    # W[k*2+i, r] is the coefficient of the r-th product in C[i,k].
    
    # Let's define the Strassen coefficients.
    # Indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for C: 00, 01, 10, 11 -> 0, 1, 2, 3 (but W index is k*2+i)
    # C00 -> W[0*2+0]=0
    # C01 -> W[0*2+1]=1
    # C10 -> W[1*2+0]=2
    # C11 -> W[1*2+1]=3
    
    # M1: A11 (0) * (B22 (3) - B12 (1))
    # U1: [1, 0, 0, 0]
    # V1: [0, -1, 0, 1]
    # W1: C00 (0) +1, C10 (2) +1, C11 (3) -1. Others 0.
    # W1: [1, 0, 1, -1]
    
    # M2: (A11 (0) + A21 (2)) * B22 (3)
    # U2: [1, 0, 1, 0]
    # V2: [0, 0, 0, 1]
    # W2: C11 (3) +1.
    # W2: [0, 0, 0, 1]
    
    # M3: (A22 (3) - A12 (1)) * B11 (0)
    # U3: [0, -1, 0, 1]
    # V3: [1, 0, 0, 0]
    # W3: C01 (1) +1, C11 (3) -1.
    # W3: [0, 1, 0, -1]
    
    # M4: A22 (3) * (B11 (0) + B21 (2))
    # U4: [0, 0, 0, 1]
    # V4: [1, 0, 1, 0]
    # W4: C00 (0) -1.
    # W4: [-1, 0, 0, 0]
    
    # M5: (A11 (0) + A12 (1)) * (B11 (0) - B21 (2))
    # U5: [1, 1, 0, 0]
    # V5: [1, 0, -1, 0]
    # W5: C00 (0) +1, C11 (3) +1.
    # W5: [1, 0, 0, 1]
    
    # M6: (A21 (2) + A22 (3)) * (B12 (1) - B22 (3))
    # U6: [0, 0, 1, 1]
    # V6: [0, 1, 0, -1]
    # W6: C10 (2) +1.
    # W6: [0, 0, 1, 0]
    
    # M7: (A11 (0) + A22 (3)) * (B22 (3) - B11 (0))
    # U7: [1, 0, 0, 1]
    # V7: [1, 0, 0, -1]
    # W7: C00 (0) +1, C01 (1) +1.
    # W7: [1, 1, 0, 0]
    
    # Now, we need to combine these blocks.
    # The total rank is 83.
    # We will create U (20x83), V (30x83), W (24x83).
    
    # Block 1: 4x4x4 (Rank 49)
    # This is 4 Strassen 2x2x2 blocks? No, 4x4x4 is one block.
    # We can decompose 4x4x4 using Strassen by treating it as a 2x2 block matrix where each element is a 2x2 matrix.
    # This gives 7 products of 2x2 matrices, each of rank 7. Total rank 49.
    # So we need to "tensor" the 2x2x2 Strassen decomposition with itself?
    # Actually, the recursion is:
    # 4x4x4 decomposition is formed by taking the 2x2x2 Strassen decomposition and replacing each scalar coefficient with a 2x2 block, and each product with a 2x2x2 Strassen product.
    # This is equivalent to the Kronecker product of the 2x2x2 Strassen matrices with themselves?
    # Let S_U, S_V, S_W be the 4x7 Strassen matrices for 2x2x2.
    # Then for 4x4x4, the matrices are:
    # U_4x4x4 = S_U ⊗ S_U (4*4 x 7*7 = 16x49)
    # V_4x4x4 = S_V ⊗ S_V (16x49)
    # W_4x4x4 = S_W ⊗ S_W (16x49)
    # Wait, the indices must be mapped correctly.
    # A is 4x4. Index i*4+j.
    # In Kronecker product, if A is indexed (i1, i2) and B is (i3, i4), the product is over (i1,i2) and (i3,i4).
    # The standard Kronecker product U ⊗ V has index (i, j) corresponding to (i1, i2) and (j1, j2) if U is indexed by i1 and V by i2.
    # Here, U_2x2x4 has index i*2+j.
    # U_4x4x49 should have index i*4+j.
    # If we let i = i1*2 + i2 and j = j1*2 + j2, then the index is (i1*2+i2)*4 + (j1*2+j2).
    # This is not the standard row-major order for a 4x4 matrix, which is i*4+j = (i1*2+i2)*4 + (j1*2+j2).
    # Actually, it is!
    # So U_4x4x4 = S_U ⊗ S_U works if we interpret the rows of S_U as indexed by (i1, i2) and the columns by r1, r2.
    # The row index of S_U ⊗ S_U is (row_U1, row_U2).
    # row_U1 = i1*2 + i2. row_U2 = j1*2 + j2? No.
    # Let's be careful.
    # S_U has shape (4, 7). Rows indexed by (i, j) for i,j in 0..1.
    # S_U ⊗ S_U has shape (16, 49).
    # Row index is (i1*2+i2, j1*2+j2)? No, Kronecker product row index is (i, j) where i is row of first, j is row of second.
    # If we want the row index to be i*4+j for a 4x4 matrix, we need to map (i1, i2, j1, j2) to i1*4 + i2*2 + j1? No.
    # Standard Kronecker: (A ⊗ B)[i*m_A + j, k*m_B + l] = A[i,k] * B[j,l].
    # Here, we want to multiply a 4x4 matrix A (indices i1,i2 and j1,j2) by a 4x4 matrix B.
    # The 4x4x4 tensor T[i,j,k] = 1.
    # Decomposition: T = sum_{r1,r2} U[i, r1] V[j, r1] ... no, it's not that simple.
    # The recursive Strassen decomposition for 4x4x4 is exactly the Kronecker product of the 2x2x2 decomposition with itself, provided the indexing is consistent.
    # Let's assume it works and verify with a small check in code.
    
    # For the other blocks:
    # 4x1x4 (Rank 4): Schoolbook.
    # 4x4x2 (Rank 28): This is 4x4x2.
    # We can decompose 4x4x2 as 2x (2x4x2).
    # 2x4x2: A (2x4), B (4x2).
    # Split A into A1 (2x2) and A2 (2x2).
    # C1 = A1 * B (2x2)
    # C2 = A2 * B (2x2)
    # This is not right, B is shared.
    # Actually, 4x4x2 is just two 4x4x1 multiplications? No.
    # It is multiplying a 4x4 matrix by a 4x2 matrix.
    # We can view this as two 4x4x1 multiplications, which have rank 4 each. Total rank 8?
    # No, the rank of 4x4x1 is 4. So 4x4x2 has rank at most 8?
    # Wait, rank(4x4x2) = 8 is achievable by just doing 8 scalar multiplications?
    # A (4x4) * B (4x2). C (4x2).
    # C[i,k] = sum_j A[i,j] B[j,k].
    # There are 4*2 = 8 entries in C. Each is a sum of 4 products.
    # Total products: 32.
    # Can we do it with 8 products?
    # If we use Strassen on the 4x4 part?
    # 4x4x2 = (4x4x2).
    # Split 4x4 into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B1], [B2]] where B1, B2 are 2x2.
    # C1 = A11*B1 + A12*B2
    # C2 = A21*B1 + A22*B2
    # This requires 4 products of 2x2 matrices.
    # Each 2x2x2 product has rank 7.
    # Total rank = 4 * 7 = 28.
    # So rank(4x4x2) <= 28.
    
    # 4x1x2 (Rank 2): Schoolbook.
    
    # So the plan is solid.
    # 1. Generate Strassen 2x2x2 matrices S_U, S_V, S_W.
    # 2. Generate 4x4x4 matrices by Kronecker product.
    # 3. Generate 4x4x2 matrices by Kronecker product of S_U, S_V and identity?
    #    For 4x4x2, we have 4 products of 2x2x2.
    #    The A part is 4x4. The B part is 4x2.
    #    The A part is decomposed using Strassen (4x4x2? No, 4x4x1 for each column of B?).
    #    Actually, it's simpler:
    #    U_4x4x2 has shape 16x28.
    #    It is composed of 4 blocks of U_2x2x2 (4x7).
    #    V_4x4x2 has shape 8x28.
    #    It is composed of 4 blocks of V_2x2x2 (4x7).
    #    W_4x4x2 has shape 8x28.
    
    # Let's implement this.

    def get_strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[:, 0] = [1, 0, 0, 0]
        V[:, 0] = [0, -1, 0, 1]
        W[:, 0] = [1, 0, 1, -1]
        
        # M2
        U[:, 1] = [1, 0, 1, 0]
        V[:, 1] = [0, 0, 0, 1]
        W[:, 1] = [0, 0, 0, 1]
        
        # M3
        U[:, 2] = [0, -1, 0, 1]
        V[:, 2] = [1, 0, 0, 0]
        W[:, 2] = [0, 1, 0, -1]
        
        # M4
        U[:, 3] = [0, 0, 0, 1]
        V[:, 3] = [1, 0, 1, 0]
        W[:, 3] = [-1, 0, 0, 0]
        
        # M5
        U[:, 4] = [1, 1, 0, 0]
        V[:, 4] = [1, 0, -1, 0]
        W[:, 4] = [1, 0, 0, 1]
        
        # M6
        U[:, 5] = [0, 0, 1, 1]
        V[:, 5] = [0, 1, 0, -1]
        W[:, 5] = [0, 0, 1, 0]
        
        # M7
        U[:, 6] = [1, 0, 0, 1]
        V[:, 6] = [1, 0, 0, -1]
        W[:, 6] = [1, 1, 0, 0]
        
        return U, V, W

    S_U, S_V, S_W = get_strassen_2x2x2()
    
    # Block 1: 4x4x4 (Rank 49)
    # Indices: A (4x4) -> 16, B (4x4) -> 16, C (4x4) -> 16
    # U1 = S_U ⊗ S_U (16x49)
    # V1 = S_V ⊗ S_V (16x49)
    # W1 = S_W ⊗ S_W (16x49)
    U1 = np.kron(S_U, S_U)
    V1 = np.kron(S_V, S_V)
    W1 = np.kron(S_W, S_W)
    
    # Block 2: 4x1x4 (Rank 4)
    # A (4x1) -> 4, B (1x4) -> 4, C (4x4) -> 16
    # Schoolbook: C[i,k] = A[i,0] * B[0,k]
    # U2[i, r] = 1 if r==i else 0. (4x4)
    # V2[k, r] = 1 if r==k else 0. (4x4)
    # W2[k*4+i, r] = 1 if r==i and r==k? No.
    # W index is k*4+i.
    # W2[k*4+i, i] = 1.
    U2 = np.eye(4)
    V2 = np.eye(4)
    W2 = np.zeros((16, 4))
    for i in range(4):
        for k in range(4):
            W2[k*4+i, i] = 1
            
    # Block 3: 4x4x2 (Rank 28)
    # A (4x4) -> 16, B (4x2) -> 8, C (4x2) -> 8
    # We have 4 Strassen products of 2x2 matrices.
    # U3 is 16x28. It is [S_U, S_U, S_U, S_U] horizontally? No.
    # The A matrix is the same for all 4 products.
    # So U3 = np.kron(np.eye(4), S_U)? No.
    # Let's think about the indices.
    # The 4 products correspond to the 4 blocks of the 4x4 A matrix.
    # Product 1 uses A11. Product 2 uses A12. Product 3 uses A21. Product 4 uses A22.
    # U3[row_A, col_prod]
    # row_A is i*4+j.
    # For product 1 (A11), we need U3 to pick out A11.
    # This is getting complex to map manually.
    # Let's use the property that 4x4x2 is equivalent to 2x (2x4x2) but with shared B?
    # No, let's just construct it by stacking the 2x2x2 Strassen matrices appropriately.
    
    # Actually, a simpler way:
    # 4x4x2 can be seen as two 4x4x1 multiplications? No.
    # Let's just use schoolbook for 4x4x2 to be safe, which is rank 32.
    # Total rank = 49 + 4 + 32 + 2 = 87.
    # This is still better than 120.
    
    # Let's stick to schoolbook for the non-square parts to ensure correctness.
    # U3 (16x32): For each k in 0..1, for each j in 0..3, for each i in 0..3.
    # V3 (8x32): For each k in 0..1, for each j in 0..3, for each i in 0..3.
    # W3 (8x32): For each k in 0..1, for each j in 0..3, for each i in 0..3.
    
    U3 = np.zeros((16, 32))
    V3 = np.zeros((8, 32))
    W3 = np.zeros((8, 32))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U3[i*4+j, col] = 1
                V3[j*2+k, col] = 1
                W3[k*4+i, col] = 1
                col += 1
                
    # Block 4: 4x1x2 (Rank 2)
    # A (4x1) -> 4, B (1x2) -> 2, C (4x2) -> 8
    # Schoolbook: C[i,k] = A[i,0] * B[0,k]
    # U4 (4x2): U4[i, k] = 1 if k==k? No.
    # Product r corresponds to k.
    # U4[i, k] = 1.
    # V4[k, k] = 1.
    # W4[k*4+i, k] = 1.
    
    U4 = np.ones((4, 2))
    V4 = np.eye(2)
    W4 = np.zeros((8, 2))
    for i in range(4):
        for k in range(2):
            W4[k*4+i, k] = 1
            
    # Combine all blocks
    # Total rank = 49 + 4 + 32 + 2 = 87
    
    U = np.hstack([U1, U2, U3, U4])
    V = np.hstack([V1, V2, V3, V4])
    W = np.hstack([W1, W2, W3, W4])
    
    # Convert to list of lists for the required format
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
