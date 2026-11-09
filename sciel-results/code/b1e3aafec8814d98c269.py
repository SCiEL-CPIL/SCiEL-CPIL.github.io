"""Rewritten program for 4x4 x 4x8 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 4x4 matrix by a 4x8 matrix.
    
    Returns a dict with keys 'U', 'V', 'W' where:
    - U has shape (n*m, rank) = (16, rank)
    - V has shape (m*p, rank) = (32, rank)
    - W has shape (p*n, rank) = (32, rank)
    
    The decomposition satisfies:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Use recursive Strassen decomposition
    # 4x4 x 4x8 can be split into:
    # - Split A (4x4) into 2x2 blocks: A11, A12, A21, A22
    # - Split B (4x8) into 2x4 blocks: B11, B12, B21, B22
    # - Result C (4x8) has 2x4 blocks: C11, C12, C21, C22
    #
    # Using 7 Strassen products, each is a 2x2 x 2x4 multiplication
    # Each 2x2 x 2x4 = two 2x2 x 2x2 multiplications
    # Total rank: 7 * 2 * 7 = 98
    
    terms = _generate_strassen_4x4_4x8_terms()
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for i in range(n * m):
            U[i][c] = u[i]
        for i in range(m * p):
            V[i][c] = v[i]
        for i in range(p * n):
            W[i][c] = w[i]
    
    return {"U": U, "V": V, "W": W}


def _generate_strassen_4x4_4x8_terms():
    """
    Generate rank-1 terms for 4x4 x 4x8 multiplication using recursive Strassen.
    
    Strategy:
    - Split A (4x4) into 2x2 blocks
    - Split B (4x8) into 2x4 blocks  
    - Use 7 Strassen combinations
    - Each combination gives a 2x2 x 2x4 multiplication
    - Each 2x2 x 2x4 is decomposed as two 2x2 x 2x2 multiplications using 7 Strassen terms each
    
    Total: 7 * 2 * 7 = 98 terms
    """
    terms = []
    
    # Define the 7 Strassen linear combinations for the 4x4 x 4x8 case
    # A blocks: A11(top-left 2x2), A12(top-right 2x2), A21(bottom-left 2x2), A22(bottom-right 2x2)
    # B blocks: B11(top-left 2x4), B12(top-right 2x4), B21(bottom-left 2x4), B22(bottom-right 2x4)
    
    # The 7 Strassen products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each M, we need to decompose a 2x2 x 2x4 multiplication
    # 2x2 x 2x4 can be seen as two 2x2 x 2x2 multiplications
    
    # Define helper to generate terms for 2x2 x 2x2 Strassen
    def strassen_2x2_2x2():
        """Return 7 (u, v, w) terms for 2x2 x 2x2 multiplication."""
        result = []
        # M1 = (a+d)(e+h)
        result.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        # M2 = (c+d)e
        result.append(([0,0,1,1], [1,0,0,0], [0,1,0,-1]))
        # M3 = a(f-h)
        result.append(([1,0,0,0], [0,1,0,-1], [0,0,1,1]))
        # M4 = d(g-e)
        result.append(([0,0,0,1], [-1,0,1,0], [1,1,0,0]))
        # M5 = (a+b)h
        result.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
        # M6 = (c-a)(e+f)
        result.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
        # M7 = (b-d)(g+h)
        result.append(([0,1,0,-1], [0,0,1,1], [1,0,0,1]))
        return result
    
    # Define the 7 Strassen coefficient patterns for A-side and B-side
    # Each pattern is a 4x4 matrix (for 2x2 blocks) or 4x8 matrix (for 2x4 blocks)
    # Actually, let me think about this differently.
    
    # For the 4x4 x 4x8 case:
    # A is 4x4, indexed by (i,j) where i in {0,1,2,3}, j in {0,1,2,3}
    # B is 4x8, indexed by (j,k) where j in {0,1,2,3}, k in {0,1,2,3,4,5,6,7}
    # C is 4x8, indexed by (i,k) where i in {0,1,2,3}, k in {0,1,2,3,4,5,6,7}
    
    # Block structure:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:4], B12 = B[0:2, 4:8], B21 = B[2:4, 0:4], B22 = B[2:4, 4:8]
    # C11 = C[0:2, 0:4], C12 = C[0:2, 4:8], C21 = C[2:4, 0:4], C22 = C[2:4, 4:8]
    
    # For each Strassen product M_r, we have:
    # A_r = sum of alpha_r * A_blocks (linear combination)
    # B_r = sum of beta_r * B_blocks (linear combination)
    # M_r = A_r * B_r (2x2 x 2x4 multiplication)
    
    # Then C = sum of gamma_r * M_r (linear combination)
    
    # Let me define the coefficient vectors for each of the 7 Strassen products
    
    # For M1 = (A11+A22)(B11+B22):
    # A-side: A11 + A22
    # B-side: B11 + B22
    
    # For M2 = (A21+A22)B11:
    # A-side: A21 + A22
    # B-side: B11
    
    # For M3 = A11(B12-B22):
    # A-side: A11
    # B-side: B12 - B22
    
    # For M4 = A22(B21-B11):
    # A-side: A22
    # B-side: B21 - B11
    
    # For M5 = (A11+A12)B22:
    # A-side: A11 + A12
    # B-side: B22
    
    # For M6 = (A21-A11)(B11+B12):
    # A-side: A21 - A11
    # B-side: B11 + B12
    
    # For M7 = (A12-A22)(B21+B22):
    # A-side: A12 - A22
    # B-side: B21 + B22
    
    # Now I need to lift each 2x2 x 2x4 multiplication to the full 4x4 x 4x8 level
    
    # For a 2x2 x 2x4 multiplication:
    # Input A' is 2x2, Input B' is 2x4, Output C' is 2x4
    # This can be decomposed as two 2x2 x 2x2 multiplications
    
    # Let me create a function to generate terms for a 2x2 x 2x4 multiplication
    # given the linear combinations for A and B blocks
    
    def generate_2x2_2x4_terms(a_coeff, b_coeff):
        """
        Generate rank-1 terms for a 2x2 x 2x4 multiplication.
        
        a_coeff: 4x4 matrix representing the linear combination of A blocks
                 (rows 0-1 for top, 2-3 for bottom; cols 0-1 for left, 2-3 for right)
                 Actually, let me use a different representation.
        
        Let me think about this more carefully.
        
        For the 4x4 x 4x8 case, each Strassen product involves:
        - A linear combination of the 4 A-blocks (each 2x2)
        - A linear combination of the 4 B-blocks (each 2x4)
        
        The result is a 2x2 x 2x4 multiplication.
        
        To decompose 2x2 x 2x4, I can split the 2x4 into two 2x2 blocks.
        """
        pass
    
    # Let me take a different approach. I'll directly construct the 98 terms.
    
    # For each of the 7 Strassen products, I have a 2x2 x 2x4 multiplication.
    # I'll decompose each as two 2x2 x 2x2 multiplications.
    
    # Let me define the Strassen coefficients more explicitly.
    
    # A-block indices: 0=A11, 1=A12, 2=A21, 3=A22
    # B-block indices: 0=B11, 1=B12, 2=B21, 3=B22
    
    # Strassen products:
    # M1: A-coeffs [1,0,0,1], B-coeffs [1,0,0,1]
    # M2: A-coeffs [0,0,1,1], B-coeffs [1,0,0,0]
    # M3: A-coeffs [1,0,0,0], B-coeffs [0,1,0,-1]
    # M4: A-coeffs [0,0,0,1], B-coeffs [0,0,1,-1]
    # M5: A-coeffs [1,1,0,0], B-coeffs [0,0,0,1]
    # M6: A-coeffs [0,0,1,-1], B-coeffs [1,1,0,0]
    # M7: A-coeffs [0,1,0,-1], B-coeffs [0,0,1,1]
    
    strassen_a_coeffs = [
        [1,0,0,1],  # M1
        [0,0,1,1],  # M2
        [1,0,0,0],  # M3
        [0,0,0,1],  # M4
        [1,1,0,0],  # M5
        [0,0,1,-1], # M6
        [0,1,0,-1]  # M7
    ]
    
    strassen_b_coeffs = [
        [1,0,0,1],  # M1
        [1,0,0,0],  # M2
        [0,1,0,-1], # M3
        [0,0,1,-1], # M4
        [0,0,0,1],  # M5
        [1,1,0,0],  # M6
        [0,0,1,1]   # M7
    ]
    
    # For each Strassen product, I need to generate terms for a 2x2 x 2x4 multiplication
    # where the A-side is a linear combination of 2x2 blocks and B-side is a linear combination of 2x4 blocks.
    
    # Let me create a helper to generate the full 4x4 x 4x8 terms from a 2x2 x 2x4 decomposition.
    
    # For a 2x2 x 2x4 multiplication:
    # A' is 2x2, B' is 2x4, C' is 2x4
    # I can split B' into B'_left (2x2) and B'_right (2x2)
    # Then C'_left = A' * B'_left and C'_right = A' * B'_right
    
    # Each 2x2 x 2x2 multiplication has 7 Strassen terms.
    
    # So for each Strassen product at the 4x4 x 4x8 level, I get 7 + 7 = 14 terms.
    # Total: 7 * 14 = 98 terms.
    
    # Now I need to map the local 2x2 x 2x2 indices to the global 4x4 x 4x8 indices.
    
    # Let me define the index mappings.
    
    # For A (4x4):
    # A11: rows 0-1, cols 0-1 -> global indices (0,0), (0,1), (1,0), (1,1)
    # A12: rows 0-1, cols 2-3 -> global indices (0,2), (0,3), (1,2), (1,3)
    # A21: rows 2-3, cols 0-1 -> global indices (2,0), (2,1), (3,0), (3,1)
    # A22: rows 2-3, cols 2-3 -> global indices (2,2), (2,3), (3,2), (3,3)
    
    # For B (4x8):
    # B11: rows 0-1, cols 0-3 -> global indices (0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3)
    # B12: rows 0-1, cols 4-7 -> global indices (0,4), (0,5), (0,6), (0,7), (1,4), (1,5), (1,6), (1,7)
    # B21: rows 2-3, cols 0-3 -> global indices (2,0), (2,1), (2,2), (2,3), (3,0), (3,1), (3,2), (3,3)
    # B22: rows 2-3, cols 4-7 -> global indices (2,4), (2,5), (2,6), (2,7), (3,4), (3,5), (3,6), (3,7)
    
    # For C (4x8):
    # C11: rows 0-1, cols 0-3 -> global indices (0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3)
    # C12: rows 0-1, cols 4-7 -> global indices (0,4), (0,5), (0,6), (0,7), (1,4), (1,5), (1,6), (1,7)
    # C21: rows 2-3, cols 0-3 -> global indices (2,0), (2,1), (2,2), (2,3), (3,0), (3,1), (3,2), (3,3)
    # C22: rows 2-3, cols 4-7 -> global indices (2,4), (2,5), (2,6), (2,7), (3,4), (3,5), (3,6), (3,7)
    
    # Let me create index mapping functions.
    
    def a_block_index(block_id, local_i, local_j):
        """Map A-block index to global (i,j) index in 4x4 matrix."""
        if block_id == 0:  # A11
            return (local_i, local_j)
        elif block_id == 1:  # A12
            return (local_i, local_j + 2)
        elif block_id == 2:  # A21
            return (local_i + 2, local_j)
        elif block_id == 3:  # A22
            return (local_i + 2, local_j + 2)
    
    def b_block_index(block_id, local_i, local_j):
        """Map B-block index to global (i,j) index in 4x8 matrix."""
        if block_id == 0:  # B11
            return (local_i, local_j)
        elif block_id == 1:  # B12
            return (local_i, local_j + 4)
        elif block_id == 2:  # B21
            return (local_i + 2, local_j)
        elif block_id == 3:  # B22
            return (local_i + 2, local_j + 4)
    
    def c_block_index(block_id, local_i, local_j):
        """Map C-block index to global (i,j) index in 4x8 matrix."""
        if block_id == 0:  # C11
            return (local_i, local_j)
        elif block_id == 1:  # C12
            return (local_i, local_j + 4)
        elif block_id == 2:  # C21
            return (local_i + 2, local_j)
        elif block_id == 3:  # C22
            return (local_i + 2, local_j + 4)
    
    # Now generate the 98 terms.
    
    for s in range(7):  # For each Strassen product
        a_coeff = strassen_a_coeffs[s]
        b_coeff = strassen_b_coeffs[s]
        
        # For the A-side, I have a linear combination of 4 2x2 blocks
        # For the B-side, I have a linear combination of 4 2x4 blocks
        
        # I'll decompose the 2x2 x 2x4 multiplication as two 2x2 x 2x2 multiplications
        
        # For the left half (cols 0-1 of the 2x4):
        # B'_left is a linear combination of the left halves of the B-blocks
        # For the right half (cols 2-3 of the 2x4):
        # B'_right is a linear combination of the right halves of the B-blocks
        
        # Let me generate terms for the left half first.
        
        # For the left half, the B-blocks contribute:
        # B11_left = B11[:, 0:2], B12_left = B12[:, 0:2], B21_left = B21[:, 0:2], B22_left = B22[:, 0:2]
        
        # Similarly for the right half.
        
        # I'll use the 7 Strassen terms for 2x2 x 2x2.
        
        strassen_2x2 = strassen_2x2_2x2()
        
        # For the left half:
        for t in range(7):
            u_local, v_local, w_local = strassen_2x2[t]
            
            # u_local is for a 2x2 matrix (4 elements)
            # v_local is for a 2x2 matrix (4 elements)
            # w_local is for a 2x2 matrix (4 elements)
            
            # I need to lift these to the global 4x4 x 4x8 level.
            
            u_global = [0] * 16  # 4x4 = 16 elements
            v_global = [0] * 32  # 4x8 = 32 elements
            w_global = [0] * 32  # 4x8 = 32 elements
            
            # For the A-side:
            # u_local has 4 elements corresponding to the 2x2 matrix
            # I need to distribute these across the 4 A-blocks based on a_coeff
            
            for local_i in range(2):
                for local_j in range(2):
                    val = u_local[local_i * 2 + local_j]
                    if val != 0:
                        for block_id in range(4):
                            coeff = a_coeff[block_id]
                            if coeff != 0:
                                gi, gj = a_block_index(block_id, local_i, local_j)
                                idx = gi * 4 + gj
                                u_global[idx] += val * coeff
            
            # For the B-side (left half):
            # v_local has 4 elements corresponding to a 2x2 matrix
            # The B-blocks for the left half are the left halves of B11, B12, B21, B22
            
            for local_i in range(2):
                for local_j in range(2):
                    val = v_local[local_i * 2 + local_j]
                    if val != 0:
                        for block_id in range(4):
                            coeff = b_coeff[block_id]
                            if coeff != 0:
                                gi, gj = b_block_index(block_id, local_i, local_j)
                                idx = gi * 8 + gj
                                v_global[idx] += val * coeff
            
            # For the C-side (left half):
            # w_local has 4 elements corresponding to a 2x2 matrix
            # The C-blocks for the left half are the left halves of C11, C12, C21, C22
            
            # But wait, the C-side depends on which Strassen product we're in.
            # Actually, the C-side should be determined by the Strassen reconstruction.
            
            # Let me reconsider. The Strassen algorithm gives:
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # So for each Strassen product M_s, it contributes to certain C-blocks.
            
            # Let me define the C-coefficients for each Strassen product.
            
            # M1: C11 += 1, C22 += 1
            # M2: C11 -= 1, C21 += 1, C22 -= 1
            # M3: C12 += 1, C22 += 1
            # M4: C11 += 1, C21 += 1
            # M5: C11 -= 1, C12 += 1
            # M6: C22 += 1
            # M7: C11 += 1
            
            c_coeffs = [
                {0: 1, 3: 1},    # M1: C11, C22
                {0: -1, 2: 1, 3: -1},  # M2: C11, C21, C22
                {1: 1, 3: 1},    # M3: C12, C22
                {0: 1, 2: 1},    # M4: C11, C21
                {0: -1, 1: 1},   # M5: C11, C12
                {3: 1},          # M6: C22
                {0: 1}           # M7: C11
            ]
            
            c_coeff_dict = c_coeffs[s]
            
            for local_i in range(2):
                for local_j in range(2):
                    val = w_local[local_i * 2 + local_j]
                    if val != 0:
                        for block_id, coeff in c_coeff_dict.items():
                            if coeff != 0:
                                gi, gj = c_block_index(block_id, local_i, local_j)
                                idx = gi * 8 + gj
                                w_global[idx] += val * coeff
            
            terms.append((u_global, v_global, w_global))
        
        # For the right half:
        for t in range(7):
            u_local, v_local, w_local = strassen_2x2[t]
            
            u_global = [0] * 16
            v_global = [0] * 32
            w_global = [0] * 32
            
            # For the A-side:
            for local_i in range(2):
                for local_j in range(2):
                    val = u_local[local_i * 2 + local_j]
                    if val != 0:
                        for block_id in range(4):
                            coeff = a_coeff[block_id]
                            if coeff != 0:
                                gi, gj = a_block_index(block_id, local_i, local_j)
                                idx = gi * 4 + gj
                                u_global[idx] += val * coeff
            
            # For the B-side (right half):
            # The right halves of the B-blocks
            for local_i in range(2):
                for local_j in range(2):
                    val = v_local[local_i * 2 + local_j]
                    if val != 0:
                        for block_id in range(4):
                            coeff = b_coeff[block_id]
                            if coeff != 0:
                                gi, gj = b_block_index(block_id, local_i, local_j + 2)
                                idx = gi * 8 + gj
                                v_global[idx] += val * coeff
            
            # For the C-side (right half):
            c_coeff_dict = c_coeffs[s]
            
            for local_i in range(2):
                for local_j in range(2):
                    val = w_local[local_i * 2 + local_j]
                    if val != 0:
                        for block_id, coeff in c_coeff_dict.items():
                            if coeff != 0:
                                gi, gj = c_block_index(block_id, local_i, local_j + 2)
                                idx = gi * 8 + gj
                                w_global[idx] += val * coeff
            
            terms.append((u_global, v_global, w_global))
    
    return terms

# EVOLVE-BLOCK-END
