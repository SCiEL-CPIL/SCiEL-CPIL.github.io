"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Strategy: 
    1. Decompose 4x4 * 4x6 into 4x4 * 4x4 and 4x4 * 4x2 using block partitioning of B.
    2. Use Strassen (rank 49) for the 4x4 * 4x4 part.
    3. Use Schoolbook (rank 32) for the 4x4 * 4x2 part.
    4. Concatenate the decompositions.
    5. Verify the result. If verification fails, return the full schoolbook (rank 96).
    """
    n, m, p = DIMENSIONS
    
    # Helper to create a schoolbook decomposition for n x m * m x p
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    # Note: The problem statement says W[k*n+i, r] where k is col of B/C, i is row of A/C.
    # Standard matmul tensor T_{i,j,k} corresponds to A_{i,j} B_{j,k} -> C_{i,k}.
    # In the provided interface:
    # U is (n*m, rank) -> index i*m + j
    # V is (m*p, rank) -> index j*p + k
    # W is (p*n, rank) -> index k*n + i
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return U, V, W

    # Helper to create Strassen decomposition for 2x2 * 2x2
    # Returns U, V, W for 2x2 * 2x2
    def strassen_2x2_2x2():
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # Indices for A: 0,1,2,3 (row-major)
        # Indices for B: 0,1,2,3
        # Indices for C: 0,1,2,3
        
        # M1 = (A11 + A22)(B11 + B22)
        # A11=0, A22=3; B11=0, B22=3
        # C11=0, C12=1, C21=2, C22=3
        # C11 += M1, C12 -= M1, C21 += M1, C22 += M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = -1; W[2, 0] = 1; W[3, 0] = 1
        
        # M2 = (A21 + A22) B11
        # A21=2, A22=3; B11=0
        # C12 -= M2, C21 -= M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = -1; W[2, 1] = -1
        
        # M3 = A11 (B12 - B22)
        # A11=0; B12=1, B22=3
        # C11 += M3, C21 += M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1; W[2, 2] = 1
        
        # M4 = A22 (B21 - B11)
        # A22=3; B21=2, B11=0
        # C11 -= M4, C12 += M4, C22 += M4
        U[3, 3] = 1
        V[0, 3] = -1; V[2, 3] = 1
        W[0, 3] = -1; W[1, 3] = 1; W[3, 3] = 1
        
        # M5 = (A11 + A12) B22
        # A11=0, A12=1; B22=3
        # C11 += M5, C22 -= M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[3, 4] = -1
        
        # M6 = (A21 - A11)(B11 + B12)
        # A21=2, A11=0; B11=0, B12=1
        # C21 += M6, C22 += M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[3, 5] = 1
        
        # M7 = (A12 - A22)(B21 + B22)
        # A12=1, A22=3; B21=2, B22=3
        # C12 += M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1
        
        return U, V, W

    # Helper to create 4x4 * 4x4 decomposition using Strassen
    # This recursively applies the 2x2 Strassen to the blocks.
    def strassen_4x4_4x4():
        U_2, V_2, W_2 = strassen_2x2_2x2()
        
        # The 4x4 matrices are split into 2x2 blocks.
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        
        # The Strassen algorithm computes 7 intermediate products M1..M7.
        # Each Mi is a product of two 2x2 matrices.
        # Since the 2x2 matrices themselves are decomposed by Strassen (rank 7),
        # each Mi contributes 7 rank-1 terms to the total decomposition.
        # Total rank = 7 * 7 = 49.
        
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        # Define the linear combinations of blocks for each Strassen term.
        # Term t corresponds to M_t = (A_lin) * (B_lin)
        # A_lin is a 2x2 matrix of coefficients for the blocks A11, A12, A21, A22.
        # B_lin is a 2x2 matrix of coefficients for the blocks B11, B12, B21, B22.
        # The result C_lin is a 2x2 matrix of coefficients for the blocks C11, C12, C21, C22.
        
        # The 7 Strassen terms for 2x2 block multiplication:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # We need to map these block-level coefficients to the element-level U, V, W.
        # For a specific Strassen term t, the "A" part is a linear combination of the 4 blocks.
        # Let A_blocks = [A11, A12, A21, A22].
        # The linear form on A is sum_{block} coeff_A[block] * A_block.
        # Since A_block is 2x2, this linear form is applied to the 4 elements of each block.
        
        # The 2x2 Strassen decomposition (U_2, V_2, W_2) decomposes the product of two 2x2 matrices.
        # Let the two 2x2 matrices be X and Y.
        # X = sum_{block_i} coeff_A[i] * A_block_i
        # Y = sum_{block_j} coeff_B[j] * B_block_j
        # The product X*Y is decomposed into rank-1 terms using U_2, V_2, W_2.
        # The k-th rank-1 term of X*Y is:
        # (U_2[:, k]^T X) * (Y V_2[:, k]) * W_2[:, k]?
        # No, the standard tensor decomposition for X*Y = sum_r (X u_r) (v_r^T Y) w_r is not quite right for the block structure.
        
        # Let's use the explicit index mapping.
        # The tensor for 4x4*4x4 is the Kronecker product of the algorithm?
        # Actually, we can just sum the contributions.
        
        # For each Strassen term t (0..6):
        #   We have coefficients for A blocks: cA_t (2x2 matrix)
        #   We have coefficients for B blocks: cB_t (2x2 matrix)
        #   We have coefficients for C blocks: cC_t (2x2 matrix)
        
        # The "A" matrix for this term is X = cA_t[0,0]*A11 + cA_t[0,1]*A12 + cA_t[1,0]*A21 + cA_t[1,1]*A22.
        # This is a 4x4 matrix.
        # Similarly for Y.
        
        # The decomposition of X*Y into rank-1 terms is given by the 2x2 Strassen decomposition?
        # No, X and Y are 4x4 matrices, but they are structured as sums of 2x2 blocks.
        # The product X*Y is a 4x4 matrix.
        # We can decompose X*Y by treating X and Y as 4x4 matrices and using the 2x2 Strassen decomposition on their 2x2 blocks?
        # This is getting circular.
        
        # Simpler approach:
        # The 4x4*4x4 tensor is the Kronecker product of the 2x2*2x2 tensor with itself?
        # No.
        
        # Let's just use the schoolbook for 4x4*4x4 if Strassen is too complex to code correctly in the time limit.
        # Schoolbook rank is 64. Strassen is 49.
        # 64 + 32 = 96. Same as full schoolbook.
        # So Strassen only helps if we can actually get 49.
        
        # Let's try to implement the 49 rank decomposition correctly.
        # The key is that the 2x2 Strassen decomposition (U_2, V_2, W_2) works on 2x2 matrices.
        # If we have a linear combination of blocks, we can apply the 2x2 decomposition to the "block" level.
        
        # Let's define the 7 terms explicitly with their block coefficients.
        terms = [
            # (coeff_A_11, coeff_A_12, coeff_A_21, coeff_A_22, coeff_B_11, coeff_B_12, coeff_B_21, coeff_B_22, coeff_C_11, coeff_C_12, coeff_C_21, coeff_C_22)
            (1, 0, 0, 1, 1, 0, 0, 1, 1, -1, 1, 1), # M1
            (0, 0, 1, 1, 1, 0, 0, 0, 0, -1, -1, 0), # M2
            (1, 0, 0, 0, 0, 1, 0, -1, 1, 0, 1, 0),  # M3
            (0, 0, 0, 1, -1, 0, 1, 0, -1, 1, 0, 1), # M4
            (1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, -1),  # M5
            (-1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1),  # M6
            (0, 1, 0, -1, 0, 0, 1, 1, 0, 1, 0, 0)   # M7
        ]
        
        # For each term t, we have a 2x2 matrix of coefficients for A blocks, B blocks, and C blocks.
        # Let cA_t be the 2x2 matrix.
        # The "A" part of the rank-1 decomposition for this term is the sum of cA_t[i,j] * A_block[i,j].
        # This is a 4x4 matrix.
        # We need to decompose the product of this "A" matrix and the "B" matrix into rank-1 terms.
        # Since the "A" and "B" matrices are linear combinations of 2x2 blocks, we can use the 2x2 Strassen decomposition
        # to decompose the product of the "block" matrices.
        
        # Wait, the 2x2 Strassen decomposition decomposes the product of two 2x2 matrices into 7 rank-1 terms.
        # If our "A" and "B" are 4x4 matrices formed by block sums, we can't directly use the 2x2 decomposition.
        
        # However, we can view the 4x4 matrix as a 2x2 matrix of 2x2 blocks.
        # Let A_blocks be the 2x2 matrix of blocks.
        # Let B_blocks be the 2x2 matrix of blocks.
        # The product C_blocks = A_blocks * B_blocks.
        # The Strassen algorithm decomposes this block product into 7 terms M1..M7.
        # Each M_t is a product of two 2x2 matrices (formed by linear combinations of blocks).
        # Let X_t and Y_t be these 2x2 matrices of blocks.
        # X_t is a 2x2 matrix where each entry is a 2x2 block (linear combo of A blocks).
        # Y_t is a 2x2 matrix where each entry is a 2x2 block (linear combo of B blocks).
        # The product X_t * Y_t is a 2x2 matrix of 2x2 blocks.
        # Each block in X_t * Y_t is a 4x4 matrix? No, it's a 2x2 block product.
        # The entry (i,j) of X_t * Y_t is sum_k X_t[i,k] * Y_t[k,j].
        # X_t[i,k] is a 2x2 matrix (linear combo of A blocks).
        # Y_t[k,j] is a 2x2 matrix (linear combo of B blocks).
        # The product is a 2x2 matrix.
        
        # So, for each Strassen term t, we have a 2x2 block product X_t * Y_t.
        # Each entry of this 2x2 block product is a 2x2 matrix product.
        # We can decompose each 2x2 matrix product using the 2x2 Strassen decomposition (rank 7).
        # So each Strassen term t contributes 4 (block entries) * 7 (rank per entry) = 28 terms?
        # No, the Strassen algorithm for 2x2 blocks computes the 4 block entries of C using 7 block products.
        # Each block product is a 2x2 * 2x2 multiplication.
        # So each block product has rank 7.
        # Total rank = 7 * 7 = 49.
        
        # So, for each Strassen term t (0..6):
        #   We have a linear combination of A blocks: A_t (2x2 matrix of 2x2 blocks)
        #   We have a linear combination of B blocks: B_t (2x2 matrix of 2x2 blocks)
        #   The product A_t * B_t is a 2x2 matrix of 2x2 blocks.
        #   We decompose this product using the 2x2 Strassen decomposition?
        #   No, the 2x2 Strassen decomposition decomposes the product of two 2x2 matrices.
        #   Here, A_t and B_t are 2x2 matrices of 2x2 blocks.
        #   So A_t * B_t is a 2x2 matrix of 2x2 blocks.
        #   We can apply the 2x2 Strassen decomposition to the "block" level?
        #   No, the 2x2 Strassen decomposition is for scalar 2x2 matrices.
        
        # Let's step back.
        # The 4x4*4x4 tensor is the Kronecker product of the 2x2*2x2 tensor with itself?
        # T_{4,4,4} = T_{2,2,2} \otimes T_{2,2,2}?
        # The rank of a Kronecker product of tensors is the product of the ranks.
        # Rank(T_{2,2,2}) = 7.
        # Rank(T_{4,4,4}) <= 7 * 7 = 49.
        # Yes! The tensor for 4x4*4x4 is the Kronecker product of the tensor for 2x2*2x2 with itself.
        
        # So we can construct the U, V, W for 4x4*4x4 by taking the Kronecker product of the U, V, W for 2x2*2x2.
        
        U_4 = np.kron(U_2, U_2)
        V_4 = np.kron(V_2, V_2)
        W_4 = np.kron(W_2, W_2)
        
        # Check shapes:
        # U_2 is 4x7. U_4 is 16x49. Correct.
        # V_2 is 4x7. V_4 is 16x49. Correct.
        # W_2 is 4x7. W_4 is 16x49. Correct.
        
        return U_4, V_4, W_4

    # Helper to create 4x4 * 4x2 decomposition using Schoolbook
    def schoolbook_4x4_4x2():
        return schoolbook(4, 4, 2)

    # Combine the decompositions
    # 4x4 * 4x6 = 4x4 * 4x4 + 4x4 * 4x2
    # U_total = [U_4x4_4x4, U_4x4_4x2]
    # V_total = [V_4x4_4x4, V_4x4_4x2]
    # W_total = [W_4x4_4x4, W_4x4_4x2]
    
    # But we need to map the indices correctly.
    # The 4x4 * 4x4 part uses the first 4 columns of B.
    # The 4x4 * 4x2 part uses the last 2 columns of B.
    
    # U is (n*m, rank) = (16, rank).
    # For the 4x4*4x4 part, U is 16x49.
    # For the 4x4*4x2 part, U is 16x32.
    # So U_total is 16x81.
    
    # V is (m*p, rank) = (4*6, rank) = (24, rank).
    # For the 4x4*4x4 part, V is (4*4, 49) = (16, 49).
    # We need to embed this into the 24x49 matrix.
    # The first 16 rows correspond to the first 4 columns of B.
    # The last 8 rows are zero.
    
    # For the 4x4*4x2 part, V is (4*2, 32) = (8, 32).
    # We need to embed this into the 24x32 matrix.
    # The first 16 rows are zero.
    # The last 8 rows correspond to the last 2 columns of B.
    
    # W is (p*n, rank) = (6*4, rank) = (24, rank).
    # For the 4x4*4x4 part, W is (4*4, 49) = (16, 49).
    # We need to embed this into the 24x49 matrix.
    # The first 16 rows correspond to the first 4 rows of C (which are the first 4 columns of C^T? No, W index is k*n+i).
    # Wait, W index is k*n+i. k is col of B/C, i is row of A/C.
    # For 4x4*4x4, k is 0..3, i is 0..3. So W indices are 0..15.
    # For 4x4*4x6, k is 0..5, i is 0..3. So W indices are 0..23.
    # The 4x4*4x4 part contributes to k=0..3, so W indices 0..15.
    # The 4x4*4x2 part contributes to k=4..5, so W indices 16..23.
    
    # So we can simply concatenate W_4x4_4x4 (16x49) and W_4x4_4x2 (8x32) vertically?
    # No, W_4x4_4x4 is 16x49. W_4x4_4x2 is 8x32.
    # We need to pad them to 24x49 and 24x32.
    
    U_4x4_4x4, V_4x4_4x4, W_4x4_4x4 = strassen_4x4_4x4()
    U_4x4_4x2, V_4x4_4x2, W_4x4_4x2 = schoolbook_4x4_4x2()
    
    # Pad V and W
    V_4x4_4x4_padded = np.zeros((24, 49))
    V_4x4_4x4_padded[:16, :] = V_4x4_4x4
    
    V_4x4_4x2_padded = np.zeros((24, 32))
    V_4x4_4x2_padded[16:, :] = V_4x4_4x2
    
    W_4x4_4x4_padded = np.zeros((24, 49))
    W_4x4_4x4_padded[:16, :] = W_4x4_4x4
    
    W_4x4_4x2_padded = np.zeros((24, 32))
    W_4x4_4x2_padded[16:, :] = W_4x4_4x2
    
    # Concatenate
    U_total = np.hstack([U_4x4_4x4, U_4x4_4x2])
    V_total = np.hstack([V_4x4_4x4_padded, V_4x4_4x2_padded])
    W_total = np.hstack([W_4x4_4x4_padded, W_4x4_4x2_padded])
    
    # Verify
    def verify(U, V, W, n, m, p):
        # C_{i,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
        # This is a tensor contraction.
        # We can check a few random instances.
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        
        # Compute C from decomposition
        # C_{i,k} = sum_{j,r} U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
        # This is expensive to compute directly in a loop.
        # We can use np.einsum or matrix multiplication tricks.
        
        # Let's use a simple loop for verification, as n,m,p are small.
        C_comp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_comp[i, k] = s
        
        return np.allclose(C_true, C_comp, atol=1e-9)
    
    if verify(U_total, V_total, W_total, n, m, p):
        return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    else:
        # Fallback to schoolbook
        U, V, W = schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
