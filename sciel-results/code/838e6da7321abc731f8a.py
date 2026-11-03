import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: M_{4,4,4} using Strassen
    # We will generate the decomposition for 4x4 * 4x4
    # Rank should be 49
    
    # Part 2: M_{4,4,1} using Schoolbook
    # Rank should be 16
    
    # Total Rank 65
    
    # We need to build U, V, W matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # Note: The task uses W[k*n+i, r] for the output factor.
    # Standard tensor decomposition: C_{ik} = sum_r U_{ir} V_{rk} W_{ki} ?
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # And U is (n*m, rank), V is (m*p, rank).
    # Usually, C = A B.
    # A is n x m. B is m x p. C is n x p.
    # Index of A: i (0..n-1), j (0..m-1). Flat index i*m + j.
    # Index of B: j (0..m-1), k (0..p-1). Flat index j*p + k.
    # Index of C: i (0..n-1), k (0..p-1). Flat index i*p + k.
    # The task says W is (p*n, rank) and W[k*n+i, r].
    # This corresponds to C_{ik} being at index k*n + i.
    # So the output is stored in column-major order relative to the (p, n) grid?
    # Or just a specific flattening.
    # Let's check the seed code:
    # W[k * n + i][column] = 1
    # So yes, W index is k*n + i.
    
    # We will build the decomposition for M_{4,4,4} and M_{4,4,1} separately and concatenate.
    
    def get_strassen_2x2():
        # Returns U, V, W for 2x2 * 2x2
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # Strassen formulas:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # Indices for 2x2:
        # A: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (W index k*n+i = k*2+i)
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1; W[1, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[3, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[0, 5] = 1; W[2, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1; W[3, 6] = 1
        
        return U, V, W

    # Get the base 2x2 decomposition
    U2, V2, W2 = get_strassen_2x2()
    
    # Now build 4x4 decomposition
    # We have 7 block products.
    # Each block product is 2x2 * 2x2.
    # We need to map the block indices to global indices.
    
    # Global A indices:
    # A11: 0,1,4,5 (rows 0-1, cols 0-1)
    # A12: 2,3,6,7 (rows 0-1, cols 2-3)
    # A21: 8,9,12,13 (rows 2-3, cols 0-1)
    # A22: 10,11,14,15 (rows 2-3, cols 2-3)
    
    # Wait, A is 4x4. Flat index i*m+j.
    # i=0: 0,1,2,3
    # i=1: 4,5,6,7
    # i=2: 8,9,10,11
    # i=3: 12,13,14,15
    
    # A11 (i=0,1; j=0,1): 0,1,4,5
    # A12 (i=0,1; j=2,3): 2,3,6,7
    # A21 (i=2,3; j=0,1): 8,9,12,13
    # A22 (i=2,3; j=2,3): 10,11,14,15
    
    # Global B indices:
    # B is 4x4. Flat index j*p+k.
    # j=0: 0,1,2,3
    # j=1: 4,5,6,7
    # j=2: 8,9,10,11
    # j=3: 12,13,14,15
    
    # B11 (j=0,1; k=0,1): 0,1,4,5
    # B12 (j=0,1; k=2,3): 2,3,6,7
    # B21 (j=2,3; k=0,1): 8,9,12,13
    # B22 (j=2,3; k=2,3): 10,11,14,15
    
    # Global C indices (W):
    # C is 4x4. Flat index k*n+i.
    # k=0: 0,1,2,3 (i=0,1,2,3)
    # k=1: 4,5,6,7
    # k=2: 8,9,10,11
    # k=3: 12,13,14,15
    
    # C11 (k=0,1; i=0,1): 0,1,4,5
    # C12 (k=2,3; i=0,1): 8,9,12,13
    # C21 (k=0,1; i=2,3): 2,3,6,7
    # C22 (k=2,3; i=2,3): 10,11,14,15
    
    # We will create U4, V4, W4 of shape (16, 49), (16, 49), (16, 49)
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # Strassen Block Definitions
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C12 += M1, C21 += M1, C22 += M1
    # M2 = (A21+A22)B11 -> C21 += M2, C22 += M2
    # M3 = A11(B12-B22) -> C11 += M3, C12 += M3
    # M4 = A22(B21-B11) -> C21 += M4, C22 += M4
    # M5 = (A11+A12)B22 -> C12 += M5, C22 += M5
    # M6 = (A21-A11)(B11+B12) -> C11 += M6, C21 += M6
    # M7 = (A12-A22)(B21+B22) -> C12 += M7, C22 += M7
    
    # Note: The C contributions are:
    # C11 = M1 + M3 + M6 - M2
    # C12 = M1 + M3 + M5 + M7
    # C21 = M1 + M2 + M4 - M6
    # C22 = M1 + M2 + M4 + M5 - M7
    
    # Let's verify:
    # C11 = A11 B11 + A12 B21
    # M1 + M3 + M6 - M2
    # = (A11+A22)(B11+B22) + A11(B12-B22) + (A21-A11)(B11+B12) - (A21+A22)B11
    # = A11 B11 + A11 B22 + A22 B11 + A22 B22 + A11 B12 - A11 B22 + A21 B11 + A21 B12 - A11 B11 - A11 B12 - A21 B11 - A22 B11
    # = A11 B11 + A21 B12 + A22 B22 - A22 B11 ? No.
    # Let's re-evaluate M6 and M2.
    # M6 = (A21-A11)(B11+B12) = A21 B11 + A21 B12 - A11 B11 - A11 B12
    # M2 = (A21+A22)B11 = A21 B11 + A22 B11
    # M6 - M2 = A21 B12 - A11 B12 - A22 B11
    # M1 + M3 = (A11+A22)(B11+B22) + A11(B12-B22)
    # = A11 B11 + A11 B22 + A22 B11 + A22 B22 + A11 B12 - A11 B22
    # = A11 B11 + A22 B11 + A22 B22 + A11 B12
    # Sum = A11 B11 + A22 B11 + A22 B22 + A11 B12 + A21 B12 - A11 B12 - A22 B11
    # = A11 B11 + A22 B22 + A21 B12
    # This is not C11.
    
    # I made a mistake in the Strassen formulas or the C combination.
    # Standard Strassen:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 + P4 - P2 - P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P3 + P5 + P7
    
    # Let's check C11:
    # P1 + P4 - P2 - P7
    # P1 = A11 B11 + A11 B22 + A22 B11 + A22 B22
    # P4 = A22 B21 - A22 B11
    # P2 = A21 B11 + A22 B11
    # P7 = A12 B21 + A12 B22 - A22 B21 - A22 B22
    # Sum = A11 B11 + A11 B22 + A22 B11 + A22 B22 + A22 B21 - A22 B11 - A21 B11 - A22 B11 - A12 B21 - A12 B22 + A22 B21 + A22 B22
    # = A11 B11 + A11 B22 + A22 B21 + A22 B21 - A21 B11 - A12 B21 - A12 B22 + A22 B22
    # This is getting messy. I will rely on the known correct Strassen formulas for the C blocks.
    
    # Correct C blocks for Strassen:
    # C11 = P1 + P4 - P2 - P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P3 + P5 + P7
    
    # I will use these coefficients to populate W4.
    
    # For each Pi, we have a 2x2 decomposition (7 mults).
    # We need to map the local indices of Pi to global indices.
    
    # Local indices for Pi (2x2):
    # U: 0,1,2,3
    # V: 0,1,2,3
    # W: 0,1,2,3
    
    # Global mapping for A blocks:
    # A11: [0,1,4,5]
    # A12: [2,3,6,7]
    # A21: [8,9,12,13]
    # A22: [10,11,14,15]
    
    # Global mapping for B blocks:
    # B11: [0,1,4,5]
    # B12: [2,3,6,7]
    # B21: [8,9,12,13]
    # B22: [10,11,14,15]
    
    # Global mapping for C blocks:
    # C11: [0,1,4,5] (k=0,1; i=0,1)
    # C12: [8,9,12,13] (k=2,3; i=0,1)
    # C21: [2,3,6,7] (k=0,1; i=2,3)
    # C22: [10,11,14,15] (k=2,3; i=2,3)
    
    # We will iterate through the 7 Pi's.
    # For each Pi, we add its U, V, W contributions to U4, V4, W4.
    
    # Define the block indices for each Pi
    # Pi = Ai * Bj
    # A_indices: list of 4 global indices
    # B_indices: list of 4 global indices
    
    # P1: (A11+A22)(B11+B22)
    # This is a sum of two products.
    # (A11+A22) is a block. (B11+B22) is a block.
    # We can treat this as a single 2x2 multiplication where the inputs are linear combinations.
    # However, the U, V matrices for the 2x2 Strassen decomposition assume the inputs are the blocks themselves.
    # If we pass (A11+A22) as the "A11" block, we need to map the local indices of this new block to global indices.
    # Local index 0 (A11 of the sub-problem) corresponds to A11 and A22 in the global problem.
    # So U[local_0] should be 1 for global A11 indices and 1 for global A22 indices.
    
    # This approach works.
    
    def add_block_product(u_block, v_block, w_block, coeff):
        # u_block: list of 4 global indices for the "A" block
        # v_block: list of 4 global indices for the "B" block
        # w_block: list of 4 global indices for the "C" block
        # coeff: multiplier for this block product (1 or -1)
        
        # Get the base 2x2 decomposition
        u_base, v_base, w_base = get_strassen_2x2()
        
        # The base decomposition has 7 columns.
        for col in range(7):
            # Add to U4
            for local_i in range(4):
                if u_base[local_i, col] != 0:
                    for global_i in u_block:
                        U4[global_i, col + offset] += u_base[local_i, col] * coeff
            # Add to V4
            for local_j in range(4):
                if v_base[local_j, col] != 0:
                    for global_j in v_block:
                        V4[global_j, col + offset] += v_base[local_j, col] * coeff
            # Add to W4
            for local_k in range(4):
                if w_base[local_k, col] != 0:
                    for global_k in w_block:
                        W4[global_k, col + offset] += w_base[local_k, col] * coeff

    # We need to be careful with the offset.
    # Each Pi uses 7 columns.
    # P1: 0-6
    # P2: 7-13
    # ...
    
    # Let's implement the specific block indices.
    
    A11 = [0, 1, 4, 5]
    A12 = [2, 3, 6, 7]
    A21 = [8, 9, 12, 13]
    A22 = [10, 11, 14, 15]
    
    B11 = [0, 1, 4, 5]
    B12 = [2, 3, 6, 7]
    B21 = [8, 9, 12, 13]
    B22 = [10, 11, 14, 15]
    
    C11 = [0, 1, 4, 5]
    C12 = [8, 9, 12, 13]
    C21 = [2, 3, 6, 7]
    C22 = [10, 11, 14, 15]
    
    # P1: (A11+A22)(B11+B22)
    # A_block = A11 + A22. Local 0->A11, 1->A12, 2->A21, 3->A22.
    # Wait, the 2x2 Strassen decomposition uses local indices 0,1,2,3 corresponding to A11, A12, A21, A22.
    # If we want to compute (A11+A22) * (B11+B22), we can define a virtual block where:
    # Virtual A11 = A11 + A22
    # Virtual A12 = 0
    # Virtual A21 = 0
    # Virtual A22 = 0
    # This is not right. Strassen uses the whole 2x2 block.
    
    # Actually, Strassen's algorithm computes 7 products of 2x2 blocks.
    # Each product is of the form (Linear Combo of A blocks) * (Linear Combo of B blocks).
    # We can just map the global indices directly.
    
    # For P1:
    # A_part = A11 + A22.
    # In the 2x2 Strassen decomposition, the "A" matrix is 2x2.
    # If we set the input "A" to be the block (A11+A22) in the top-left and 0 elsewhere?
    # No, the Strassen decomposition multiplies two 2x2 matrices.
    # If we want to compute X*Y where X = A11+A22 (2x2) and Y = B11+B22 (2x2),
    # we can use the 2x2 Strassen decomposition on X and Y.
    # The local indices of X are:
    # X11 = A11 + A22 (elements 0,1,4,5 and 10,11,14,15)
    # X12 = 0
    # X21 = 0
    # X22 = 0
    # This is not a 2x2 matrix multiplication in the usual sense.
    # X is a 2x2 matrix. Its elements are sums of global elements.
    # We can just add the contributions.
    
    # Let's refine the `add_block_product` function.
    # It should take the global indices for the 4 elements of the A-block, B-block, and C-block.
    # For P1, the A-block is not a single global block, but a sum.
    # So for local index 0 (which corresponds to A11 in the sub-problem),
    # we add contributions from global A11 and global A22.
    # For local index 1 (A12), we add nothing.
    # For local index 2 (A21), we add nothing.
    # For local index 3 (A22), we add nothing.
    
    # Wait, if X = A11+A22, then X is a 2x2 matrix.
    # X11 = A11_11 + A22_11
    # X12 = A11_12 + A22_12
    # X21 = A11_21 + A22_21
    # X22 = A11_22 + A22_22
    # So local index 0 (X11) corresponds to global A11_11 and A22_11.
    # Global A11_11 is index 0. Global A22_11 is index 10.
    # So for local 0, we add U[global 0] and U[global 10].
    
    # This works.
    
    # Let's implement this logic.
    
    # I will write the code now.

