import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Rank-49 decomposition for 4x4x4 matrix multiplication
    # This is a known minimal decomposition
    # We'll construct it using the block Strassen approach
    
    # For 4x4x4, we can use the following approach:
    # Split each 4x4 matrix into 2x2 blocks
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = A*B = [[C11, C12], [C21, C22]]
    
    # Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4x4, we apply Strassen to the 2x2 block structure.
    # This gives rank 7*7 = 49 for the 4x4x4 case.
    
    # Let's construct the U, V, W matrices for rank 49.
    # The indices for 4x4x4:
    # U: (4*4, 49) = (16, 49)
    # V: (4*4, 49) = (16, 49)
    # W: (4*4, 49) = (16, 49)
    
    # We'll build this systematically.
    # Each Strassen term M_k corresponds to a rank-1 decomposition of a 2x2x2 subproblem.
    # For the 4x4x4 case, we have 7 outer Strassen terms, each requiring 7 inner Strassen terms.
    
    # Let's define the 7 Strassen coefficients for the outer level.
    # Outer Strassen:
    # A blocks: A11, A12, A21, A22 (each 2x2)
    # B blocks: B11, B12, B21, B22 (each 2x2)
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each M_k, we compute a 2x2 result using Strassen, which requires 7 rank-1 terms.
    
    # Let's construct U, V, W for the 4x4x4 case.
    # We'll use a systematic approach.
    
    # Define the 7 outer Strassen coefficients as linear combinations of A and B blocks.
    # Each coefficient is a 2x2 matrix.
    
    # For U (A side):
    # Row index: i*4 + j where i is row of A (0-3), j is col of A (0-3)
    # For V (B side):
    # Row index: j*4 + k where j is row of B (0-3), k is col of B (0-3)
    # For W (C side):
    # Row index: k*4 + i where k is col of C (0-3), i is row of C (0-3)
    
    # Let's build the decomposition.
    # We'll create arrays for U, V, W of shape (16, 49).
    
    U444 = np.zeros((16, 49))
    V444 = np.zeros((16, 49))
    W444 = np.zeros((16, 49))
    
    # The 7 outer Strassen terms
    # For each term, we have 7 inner Strassen terms.
    
    # Define the outer Strassen coefficients
    # A coefficients for each of the 7 terms (each is a 4x4 matrix of coefficients)
    # B coefficients for each of the 7 terms (each is a 4x4 matrix of coefficients)
    
    # Let's define the 7 outer terms:
    # Term 0: M1 = (A11+A22)(B11+B22)
    # Term 1: M2 = (A21+A22)B11
    # Term 2: M3 = A11(B12-B22)
    # Term 3: M4 = A22(B21-B11)
    # Term 4: M5 = (A11+A12)B22
    # Term 5: M6 = (A21-A11)(B11+B12)
    # Term 6: M7 = (A12-A22)(B21+B22)
    
    # For each term, we compute a 2x2 product using Strassen, which requires 7 rank-1 terms.
    
    # Let's define the A and B block coefficients for each outer term.
    # A blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3), A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    
    # For U (A side), we need to express each A block as a linear combination.
    # For V (B side), we need to express each B block as a linear combination.
    
    # Let's create helper functions to add Strassen terms.
    
    def add_strassen_term(U, V, W, a_coeff, b_coeff, c_coeff, offset):
        """
        Add a Strassen term for a 2x2x2 subproblem.
        a_coeff: 4x4 matrix of coefficients for A
        b_coeff: 4x4 matrix of coefficients for B
        c_coeff: 4x4 matrix of coefficients for C
        offset: column offset in U, V, W
        """
        # Strassen for 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each of the 7 Strassen terms, we add a rank-1 decomposition.
        # Each term is a product of two 2x2 matrices.
        
        # Term 1: (A11+A22)(B11+B22)
        # A part: A11 + A22
        # B part: B11 + B22
        # C part: contributes to C11, C22
        
        # Let's define the 7 inner Strassen terms.
        # Each term has an A coefficient, B coefficient, and C coefficient.
        
        # For a 2x2 matrix, the blocks are:
        # A11: rows 0-1, cols 0-1 -> indices 0,1,4,5 in 4x4
        # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7 in 4x4
        # A21: rows 2-3, cols 0-1 -> indices 8,9,12,13 in 4x4
        # A22: rows 2-3, cols 2-3 -> indices 10,11,14,15 in 4x4
        
        # Actually, let's use a simpler approach.
        # We'll directly construct the U, V, W matrices by iterating over the 49 terms.
        
        pass
    
    # Given the complexity, let's use a different approach.
    # We'll use the known fact that the rank is 49 and construct a valid decomposition.
    
    # Actually, let's use the schoolbook decomposition as a fallback and try to optimize.
    # But the knowledge says rank 49 works.
    
    # Let's construct the rank-49 decomposition using the block Strassen method.
    
    # For the 4x4x4 case, we split into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each block is 2x2.
    # B = [[B11, B12], [B21, B22]] where each block is 2x2.
    # C = A*B = [[C11, C12], [C21, C22]] where each block is 2x2.
    
    # Strassen for 2x2 blocks:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M_k is a 2x2 matrix computed from 2x2 block products.
    # Each 2x2 product can be computed with Strassen (rank 7).
    # So total rank = 7 * 7 = 49.
    
    # Let's construct U, V, W for this.
    
    # For U (A side), row index i*4+j where i is row in 0-3, j is col in 0-3.
    # For V (B side), row index j*4+k where j is row in 0-3, k is col in 0-3.
    # For W (C side), row index k*4+i where k is col in 0-3, i is row in 0-3.
    
    # We'll iterate over the 7 outer Strassen terms and 7 inner Strassen terms.
    
    # Define the outer Strassen coefficients for A and B.
    # Each outer term is a product of two linear combinations of A blocks and B blocks.
    
    # Let's define the 7 outer terms as pairs of (A_coeff, B_coeff) where each is a 4x4 matrix.
    
    # Term 0: (A11+A22) and (B11+B22)
    # A11: rows 0-1, cols 0-1
    # A22: rows 2-3, cols 2-3
    # B11: rows 0-1, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # Let's create the coefficient matrices.
    
    # For A11+A22:
    # Rows 0-1, cols 0-1: 1
    # Rows 2-3, cols 2-3: 1
    # Others: 0
    
    # For B11+B22:
    # Rows 0-1, cols 0-1: 1
    # Rows 2-3, cols 2-3: 1
    # Others: 0
    
    # Let's define a helper to create a 4x4 matrix with 1s in a block.
    
    def block_matrix(row_block, col_block):
        """Create a 4x4 matrix with 1s in the specified 2x2 block."""
        M = np.zeros((4, 4))
        M[row_block[0]:row_block[1]+1, col_block[0]:col_block[1]+1] = 1
        return M
    
    # Define the 7 outer Strassen terms.
    # Each term has an A coefficient matrix and a B coefficient matrix.
    
    # A blocks:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B blocks:
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    A11 = block_matrix((0,1), (0,1))
    A12 = block_matrix((0,1), (2,3))
    A21 = block_matrix((2,3), (0,1))
    A22 = block_matrix((2,3), (2,3))
    
    B11 = block_matrix((0,1), (0,1))
    B12 = block_matrix((0,1), (2,3))
    B21 = block_matrix((2,3), (0,1))
    B22 = block_matrix((2,3), (2,3))
    
    # Outer Strassen terms:
    # Term 0: (A11+A22) and (B11+B22)
    # Term 1: (A21+A22) and B11
    # Term 2: A11 and (B12-B22)
    # Term 3: A22 and (B21-B11)
    # Term 4: (A11+A12) and B22
    # Term 5: (A21-A11) and (B11+B12)
    # Term 6: (A12-A22) and (B21+B22)
    
    outer_terms = [
        (A11 + A22, B11 + B22),  # Term 0
        (A21 + A22, B11),        # Term 1
        (A11, B12 - B22),        # Term 2
        (A22, B21 - B11),        # Term 3
        (A11 + A12, B22),        # Term 4
        (A21 - A11, B11 + B12),  # Term 5
        (A12 - A22, B21 + B22),  # Term 6
    ]
    
    # For each outer term, we compute a 2x2 product using Strassen (rank 7).
    # Each inner Strassen term is a rank-1 decomposition of a 2x2x2 subproblem.
    
    # For a 2x2x2 subproblem with A_sub and B_sub, the Strassen decomposition is:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's define the inner Strassen terms for a 2x2x2 subproblem.
    # Each term has an A coefficient (2x2), B coefficient (2x2), and C coefficient (2x2).
    
    # For a 2x2 matrix, the blocks are:
    # A11: (0,0), A12: (0,1), A21: (1,0), A22: (1,1)
    
    # Let's create 2x2 block matrices.
    
    def block_matrix_2x2(row, col):
        """Create a 2x2 matrix with 1 at (row, col)."""
        M = np.zeros((2, 2))
        M[row, col] = 1
        return M
    
    A11_2x2 = block_matrix_2x2(0, 0)
    A12_2x2 = block_matrix_2x2(0, 1)
    A21_2x2 = block_matrix_2x2(1, 0)
    A22_2x2 = block_matrix_2x2(1, 1)
    
    B11_2x2 = block_matrix_2x2(0, 0)
    B12_2x2 = block_matrix_2x2(0, 1)
    B21_2x2 = block_matrix_2x2(1, 0)
    B22_2x2 = block_matrix_2x2(1, 1)
    
    # Inner Strassen terms for 2x2x2:
    inner_terms = [
        (A11_2x2 + A22_2x2, B11_2x2 + B22_2x2, 1),  # M1, contributes to C11, C22
        (A21_2x2 + A22_2x2, B11_2x2, 1),            # M2, contributes to C21, C22
        (A11_2x2, B12_2x2 - B22_2x2, 1),            # M3, contributes to C12, C22
        (A22_2x2, B21_2x2 - B11_2x2, 1),            # M4, contributes to C11, C21
        (A11_2x2 + A12_2x2, B22_2x2, 1),            # M5, contributes to C11, C12
        (A21_2x2 - A11_2x2, B11_2x2 + B12_2x2, 1),  # M6, contributes to C22
        (A12_2x2 - A22_2x2, B21_2x2 + B22_2x2, 1),  # M7, contributes to C11, C22
    ]
    
    # Now, for each outer term, we need to combine it with each inner term.
    # The outer term gives a 4x4 A coefficient and a 4x4 B coefficient.
    # The inner term gives a 2x2 A coefficient, 2x2 B coefficient, and a C coefficient.
    
    # The product of the outer and inner terms gives a rank-1 decomposition.
    
    # Let's construct U, V, W.
    
    col = 0
    for outer_idx, (A_outer, B_outer) in enumerate(outer_terms):
        for inner_idx, (A_inner, B_inner, C_coeff) in enumerate(inner_terms):
            # The A coefficient for this rank-1 term is A_outer * A_inner (in some sense).
            # Actually, the outer term is a linear combination of A blocks, and the inner term is a linear combination of 2x2 sub-blocks.
            
            # Let's think of it as:
            # The outer term M_k = (linear_combination_of_A_blocks) * (linear_combination_of_B_blocks)
            # Each linear_combination_of_A_blocks is a 4x4 matrix.
            # The inner Strassen term is a rank-1 decomposition of the 2x2 product.
            
            # For the 4x4 case, the A coefficient for the rank-1 term is the outer A coefficient multiplied by the inner A coefficient (in the block sense).
            
            # Actually, the outer A coefficient is a 4x4 matrix that selects certain blocks of A.
            # The inner A coefficient is a 2x2 matrix that selects certain elements of a 2x2 block.
            
            # Let's construct the full 4x4 A coefficient by combining the outer and inner coefficients.
            
            # The outer A coefficient A_outer is a 4x4 matrix.
            # The inner A coefficient A_inner is a 2x2 matrix.
            
            # The full A coefficient is a 4x4 matrix where each 2x2 block is scaled by the corresponding entry of A_inner.
            
            # For example, if A_outer has 1s in the (0,0) and (2,2) blocks, and A_inner is [[1,0],[0,1]], then the full A coefficient is:
            # [[1,0,0,0],
            #  [0,1,0,0],
            #  [0,0,1,0],
            #  [0,0,0,1]]
            
            # If A_inner is [[1,1],[0,0]], then the full A coefficient is:
            # [[1,1,0,0],
            #  [0,0,0,0],
            #  [0,0,1,1],
            #  [0,0,0,0]]
            
            # Let's construct the full 4x4 A coefficient.
            
            A_full = np.zeros((4, 4))
            for i in range(2):
                for j in range(2):
                    A_full[2*i:2*i+2, 2*j:2*j+2] = A_outer[2*i:2*i+2, 2*j:2*j+2] * A_inner[i, j]
            
            B_full = np.zeros((4, 4))
            for i in range(2):
                for j in range(2):
                    B_full[2*i:2*i+2, 2*j:2*j+2] = B_outer[2*i:2*i+2, 2*j:2*j+2] * B_inner[i, j]
            
            # Now, we need to determine the C coefficient.
            # The C coefficient is determined by the inner Strassen term and the outer C combination.
            
            # For the inner Strassen term, the C coefficient is a 2x2 matrix.
            # For the outer Strassen term, the C coefficient is determined by which C blocks are affected.
            
            # Let's define the C coefficients for the inner Strassen terms.
            # M1: C11 += 1, C22 += 1
            # M2: C21 += 1, C22 -= 1
            # M3: C12 += 1, C22 += 1
            # M4: C11 += 1, C21 += 1
            # M5: C11 -= 1, C12 += 1
            # M6: C22 += 1
            # M7: C11 += 1, C22 += 1
            
            # Actually, let's use the standard Strassen C coefficients.
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # So for each inner term, the C coefficient is:
            # Term 0 (M1): C11 += 1, C22 += 1
            # Term 1 (M2): C21 += 1, C22 -= 1
            # Term 2 (M3): C12 += 1, C22 += 1
            # Term 3 (M4): C11 += 1, C21 += 1
            # Term 4 (M5): C11 -= 1, C12 += 1
            # Term 5 (M6): C22 += 1
            # Term 6 (M7): C11 += 1, C22 += 1
            
            # Let's create the C coefficient matrix for each inner term.
            # The C coefficient is a 4x4 matrix for the 4x4 case.
            
            # For the inner term, the C coefficient is a 2x2 matrix.
            # We need to map this to the 4x4 C matrix.
            
            # The 2x2 C blocks are:
            # C11: rows 0-1, cols 0-1
            # C12: rows 0-1, cols 2-3
            # C21: rows 2-3, cols 0-1
            # C22: rows 2-3, cols 2-3
            
            # Let's define the C coefficients for the 7 inner terms.
            C_inner_coeffs = [
                # M1: C11 += 1, C22 += 1
                np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]),
                # M2: C21 += 1, C22 -= 1
                np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [1, 0, -1, 0],
                          [0, 1, 0, -1]]),
                # M3: C12 += 1, C22 += 1
                np.array([[0, 0, 1, 0],
                          [0, 0, 0, 1],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]),
                # M4: C11 += 1, C21 += 1
                np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [1, 0, 0, 0],
                          [0, 1, 0, 0]]),
                # M5: C11 -= 1, C12 += 1
                np.array([[-1, 0, 1, 0],
                          [0, -1, 0, 1],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]),
                # M6: C22 += 1
                np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]),  # This is wrong, M6 only affects C22
                # M7: C11 += 1, C22 += 1
                np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]),
            ]
            
            # Fix M6: C22 += 1
            C_inner_coeffs[5] = np.array([[0, 0, 0, 0],
                                          [0, 0, 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]])
            
            # Now, the full C coefficient is the inner C coefficient scaled by the outer C combination.
            # But the outer C combination is already accounted for in the inner C coefficients.
            
            # Actually, the outer Strassen terms determine which C blocks are affected.
            # For example, Term 0 (M1) affects C11 and C22.
            # Term 1 (M2) affects C21 and C22.
            # etc.
            
            # The inner C coefficients are for the 2x2 subproblem.
            # We need to map them to the 4x4 C matrix.
            
            # Let's use the C_inner_coeffs as defined above.
            
            C_full = C_inner_coeffs[inner_idx]
            
            # Now, we have A_full, B_full, C_full for this rank-1 term.
            # A_full is 4x4, B_full is 4x4, C_full is 4x4.
            
            # U row index: i*4 + j where i is row of A, j is col of A.
            # V row index: j*4 + k where j is row of B, k is col of B.
            # W row index: k*4 + i where k is col of C, i is row of C.
            
            # For U, we add A_full[i, j] to U[i*4+j, col].
            # For V, we add B_full[j, k] to V[j*4+k, col].
            # For W, we add C_full[k, i] to W[k*4+i, col].
            
            for i in range(4):
                for j in range(4):
                    U444[i*4+j, col] = A_full[i, j]
            
            for j in range(4):
                for k in range(4):
                    V444[j*4+k, col] = B_full[j, k]
            
            for k in range(4):
                for i in range(4):
                    W444[k*4+i, col] = C_full[k, i]
            
            col += 1
    
    # Now we have U444, V444, W444 of shape (16, 49).
    
    # Verify the decomposition.
    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not verify_decomposition(U444, V444, W444, 4, 4, 4, tol=1e-7):
        # Fallback to schoolbook
        schoolbook_rank = 4 * 4 * 4
        U_school = np.zeros((16, schoolbook_rank))
        V_school = np.zeros((16, schoolbook_rank))
        W_school = np.zeros((16, schoolbook_rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_school[i*4+j, col] = 1
                    V_school[j*4+k, col] = 1
                    W_school[k*4+i, col] = 1
                    col += 1
        U444 = U_school
        V444 = V_school
        W444 = W_school
    
    # Now construct the rank-98 decomposition for 4x4x8.
    # Split B into B1 (cols 0-3) and B2 (cols 4-7).
    # Split C into C1 (cols 0-3) and C2 (cols 4-7).
    # C1 = A*B1, C2 = A*B2.
    
    R_total = 98
    U = np.zeros((16, R_total))
    V = np.zeros((32, R_total))
    W = np.zeros((32, R_total))
    
    # First block (B1, C1)
    U[:, 0:49] = U444
    V[0:16, 0:49] = V444
    W[0:16, 0:49] = W444
    
    # Second block (B2, C2)
    U[:, 49:98] = U444
    V[16:32, 49:98] = V444
    W[16:32, 49:98] = W444
    
    # Verify the full decomposition.
    if not verify_decomposition(U, V, W, 4, 4, 8, tol=1e-7):
        # Fallback to schoolbook for 4x4x8
        schoolbook_rank = 4 * 4 * 8
        U_school = np.zeros((16, schoolbook_rank))
        V_school = np.zeros((32, schoolbook_rank))
        W_school = np.zeros((32, schoolbook_rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    U_school[i*4+j, col] = 1
                    V_school[j*8+k, col] = 1
                    W_school[k*4+i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
