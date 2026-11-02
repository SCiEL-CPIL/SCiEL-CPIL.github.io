"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Generate Strassen decomposition for M_{4,4,4}
    # We will implement a recursive generator for Strassen factors
    def get_strassen_factors(size):
        """
        Returns U, V, W for M_{size, size, size} using Strassen.
        U, V, W are lists of lists.
        U: (size*size, rank)
        V: (size*size, rank)
        W: (size*size, rank)
        """
        if size == 2:
            # Base case: 2x2x2 Strassen
            # A indices: 00, 01, 10, 11
            # B indices: 00, 01, 10, 11
            # C indices: 00, 01, 10, 11
            U = [
                [1, 0, 0, 1], # M1: A00+A11
                [0, 0, 1, 1], # M2: A10+A11
                [1, 0, 0, 0], # M3: A00
                [0, 0, 0, 1], # M4: A11
                [1, 1, 0, 0], # M5: A00+A01
                [-1, 0, 1, 0],# M6: A10-A00
                [0, 1, 0, -1] # M7: A01-A11
            ]
            V = [
                [1, 0, 0, 1], # M1: B00+B11
                [1, 0, 0, 0], # M2: B00
                [0, 1, 0, -1],# M3: B01-B11
                [0, 0, 1, -1],# M4: B10-B00
                [0, 0, 0, 1], # M5: B11
                [1, 1, 0, 0], # M6: B00+B01
                [0, 0, 1, 1]  # M7: B10+B11
            ]
            W = [
                [1, 0, 0, 1], # M1 -> C00, C11
                [0, 0, 1, -1],# M2 -> C10, C11(-)
                [0, 1, 0, 1], # M3 -> C01, C11
                [1, 0, 1, 0], # M4 -> C00, C10
                [-1, 1, 0, 0],# M5 -> C00(-), C01
                [0, 0, 0, 1], # M6 -> C11
                [1, 0, -1, 0] # M7 -> C00, C10(-)
            ]
            # Note: The W vectors above are the contributions to C.
            # We must ensure the sum reconstructs C correctly.
            # The standard Strassen W vectors are:
            # W1 = [1, 0, 0, 1]
            # W2 = [0, 0, 1, -1]
            # W3 = [0, 1, 0, 1]
            # W4 = [1, 0, 1, 0]
            # W5 = [-1, 1, 0, 0]
            # W6 = [0, 0, 0, 1]
            # W7 = [1, 0, -1, 0]
            # This matches the logic derived earlier.
            return U, V, W

        # Recursive case
        half = size // 2
        U_sub, V_sub, W_sub = get_strassen_factors(half)
        rank_sub = len(U_sub)
        
        # We need to map the block indices to the full matrix indices.
        # A is size x size. Blocks are 00, 01, 10, 11.
        # Block 00 covers rows 0..half-1, cols 0..half-1.
        # Block 01 covers rows 0..half-1, cols half..size-1.
        # Block 10 covers rows half..size-1, cols 0..half-1.
        # Block 11 covers rows half..size-1, cols half..size-1.
        
        # Helper to create a sparse vector for a block
        def block_vec(block_idx, values):
            """
            values: list of length half*half
            Returns a vector of length size*size with values placed in the block.
            """
            vec = [0] * (size * size)
            row_start = (block_idx // 2) * half
            col_start = (block_idx % 2) * half
            for r in range(half):
                for c in range(half):
                    vec[(row_start + r) * size + (col_start + c)] = values[r * half + c]
            return vec

        U_full = []
        V_full = []
        W_full = []
        
        # Strassen formulas for 2x2 blocks of size 'half'
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each of the 7 outer multiplications, we decompose the resulting
        # half x half matrix multiplication using the sub-decomposition.
        
        # We need to combine the U, V vectors for the linear combinations.
        # e.g., for M1, U_part is sum of U vectors for A00 and A11.
        # But U vectors in sub-decomp are for the *entire* half x half matrix.
        # So if we want to compute (A00+A11) * (B00+B11),
        # We can't just add U vectors because the sub-decomp U vectors
        # act on the specific block.
        
        # Correct approach:
        # The tensor for M1 is T1 = T_{A00+B11, B00+B11, C_out}.
        # This is not simply T_{A00, B00} + ... because of the multiplication.
        # Wait, Strassen replaces the multiplication of the sums.
        # So M1 is a scalar multiplication of the resulting matrices?
        # No, M1 is a matrix multiplication of the sums.
        # So we need the decomposition of M_{half, half, half}.
        # Let U_sub, V_sub, W_sub be the decomposition for M_{half, half, half}.
        # Then M1 = sum_c (U_sub_c . (A00+A11)) * (V_sub_c . (B00+B11)) * W_sub_c.
        # This expands to:
        # sum_c (U_sub_c . A00 + U_sub_c . A11) * (V_sub_c . B00 + V_sub_c . B11) * W_sub_c.
        # This is NOT a valid tensor decomposition of the form sum U_i(A) V_i(B) W_i.
        # Because the term is (U.A00 + U.A11)(V.B00 + V.B11) = U.A00 V.B00 + U.A00 V.B11 + U.A11 V.B00 + U.A11 V.B11.
        # This requires 4 separate terms for each c.
        # So for each outer Strassen step, we generate 4 * rank_sub terms?
        # No, that would be 7 * 4 * 7 = 196 terms.
        # But Strassen reduces 8 mults to 7.
        # The reduction happens because we compute the product of the sums directly.
        # So we treat (A00+A11) as a single input matrix A' and (B00+B11) as B'.
        # Then we use the sub-decomposition on A', B'.
        # So we need U vectors that extract A' from A.
        # A' = A00 + A11.
        # So U'_c = U_sub_c (for A00 part) + U_sub_c (for A11 part).
        # But U_sub_c is a vector of length half*half.
        # We need to map it to the full size.
        # U_full_c = block_vec(0, U_sub_c) + block_vec(3, U_sub_c).
        
        # Let's define the linear combinations for U and V for each of the 7 steps.
        
        # Step 1: M1 = (A00+A11)(B00+B11)
        # U_lin = A00 + A11
        # V_lin = B00 + B11
        # W_lin = Identity (maps result to C00, C01, C10, C11? No, W_sub maps to the 4 blocks of the output matrix of size half x half.
        # The output of M1 is a half x half matrix.
        # This matrix contributes to C00?
        # In Strassen, the 7 products are half x half matrices.
        # C00 = M1 + M4 - M5 + M7.
        # So the result of M1 (a half x half matrix) is added to C00 (the top-left half x half block of C).
        # So W_full for this term should place the W_sub vector into the C00 block.
        
        # So for each of the 7 outer steps:
        # 1. Define U_lin: combination of block vectors for A.
        # 2. Define V_lin: combination of block vectors for B.
        # 3. For each sub-factor c:
        #    U_full = U_lin applied to A.
        #    V_full = V_lin applied to B.
        #    W_full = block_vec(output_block, W_sub_c).
        
        # Let's implement this.
        
        def combine_U(A_blocks):
            # A_blocks: dict {block_idx: coeff}
            # Returns a list of vectors (one for each sub-factor)
            # Actually, we need to return the U vectors for the full size.
            # For each sub-factor c, U_c is a vector of length size*size.
            # U_c = sum_{b, coeff} coeff * block_vec(b, U_sub_c)
            pass

        # This is getting complicated to code in the fly.
        # Given the time, I will implement the 4x4x4 decomposition by hardcoding the 49 factors
        # or using a simpler recursive structure that is easier to write.
        
        # Alternative: Just return the 49 factors generated by a known library or pre-computed?
        # No, I must generate them.
        
        # Let's try a different approach:
        # The rank of M_{4,4,4} is 49.
        # I will construct the 49 factors by iterating the 7 outer steps and 7 inner steps.
        
        U_out = []
        V_out = []
        W_out = []
        
        # Define the 7 Strassen steps
        # (U_blocks, V_blocks, W_block_idx)
        # U_blocks: {block_idx: coeff}
        # V_blocks: {block_idx: coeff}
        # W_block_idx: which output block (0,1,2,3) this result goes into.
        # Note: The result is a half x half matrix.
        # C00 gets M1, M4, M5(-), M7.
        # C01 gets M3, M5.
        # C10 gets M2, M4.
        # C11 gets M1, M2(-), M3, M6.
        
        steps = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1 -> C00, C11? No.
            # Wait, M1 is a matrix. It contributes to C00 and C11?
            # C00 = M1 + ...
            # C11 = M1 - ...
            # So M1 contributes to BOTH C00 and C11.
            # This means W for M1 must have non-zero entries in C00 block and C11 block.
            # W_sub_c is a vector of length 4 (for the half x half output).
            # When we place it in C00 block, it affects C00.
            # When we place it in C11 block, it affects C11.
            # So for M1, we need TWO sets of factors?
            # No, the tensor is sum U V W.
            # If M1 contributes to C00 and C11, then W must have components in both.
            # So W_full for M1's sub-factors will be:
            # block_vec(0, W_sub_c) + block_vec(3, W_sub_c).
            
            # Let's define the steps with their W contributions.
        ]
        
        # Step 1: M1 = (A00+A11)(B00+B11)
        # Contributes to C00 (+1) and C11 (+1)
        U_blocks = {0: 1, 3: 1}
        V_blocks = {0: 1, 3: 1}
        W_blocks = {0: 1, 3: 1}
        
        # Step 2: M2 = (A10+A11)B00
        # Contributes to C10 (+1) and C11 (-1)
        U_blocks = {2: 1, 3: 1}
        V_blocks = {0: 1}
        W_blocks = {2: 1, 3: -1}
        
        # Step 3: M3 = A00(B01-B11)
        # Contributes to C01 (+1) and C11 (+1)
        U_blocks = {0: 1}
        V_blocks = {1: 1, 3: -1}
        W_blocks = {1: 1, 3: 1}
        
        # Step 4: M4 = A11(B10-B00)
        # Contributes to C00 (+1) and C10 (+1)
        U_blocks = {3: 1}
        V_blocks = {2: 1, 0: -1}
        W_blocks = {0: 1, 2: 1}
        
        # Step 5: M5 = (A00+A01)B11
        # Contributes to C00 (-1) and C01 (+1)
        U_blocks = {0: 1, 1: 1}
        V_blocks = {3: 1}
        W_blocks = {0: -1, 1: 1}
        
        # Step 6: M6 = (A10-A00)(B00+B01)
        # Contributes to C11 (+1)
        U_blocks = {2: 1, 0: -1}
        V_blocks = {0: 1, 1: 1}
        W_blocks = {3: 1}
        
        # Step 7: M7 = (A01-A11)(B10+B11)
        # Contributes to C00 (+1) and C10 (-1)
        U_blocks = {1: 1, 3: -1}
        V_blocks = {2: 1, 3: 1}
        W_blocks = {0: 1, 2: -1}
        
        all_steps = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 2: -1})
        ]
        
        for U_b, V_b, W_b in all_steps:
            # For each sub-factor c in the half-size decomposition
            for c in range(rank_sub):
                u_vec = [0] * (size * size)
                for b, coeff in U_b.items():
                    # Add coeff * block_vec(b, U_sub[c])
                    row_start = (b // 2) * half
                    col_start = (b % 2) * half
                    for r in range(half):
                        for k in range(half):
                            u_vec[(row_start + r) * size + (col_start + k)] += coeff * U_sub[c][r * half + k]
                
                v_vec = [0] * (size * size)
                for b, coeff in V_b.items():
                    row_start = (b // 2) * half
                    col_start = (b % 2) * half
                    for r in range(half):
                        for k in range(half):
                            v_vec[(row_start + r) * size + (col_start + k)] += coeff * V_sub[c][r * half + k]
                
                w_vec = [0] * (size * size)
                for b, coeff in W_b.items():
                    row_start = (b // 2) * half
                    col_start = (b % 2) * half
                    for r in range(half):
                        for k in range(half):
                            w_vec[(row_start + r) * size + (col_start + k)] += coeff * W_sub[c][r * half + k]
                
                U_out.append(u_vec)
                V_out.append(v_vec)
                W_out.append(w_vec)
        
        return U_out, V_out, W_out

    U_444, V_444, W_444 = get_strassen_factors(4)
    
    # 2. Generate Schoolbook decomposition for M_{4,4,1}
    # A is 4x4, B is 4x1, C is 4x1.
    # Rank 16.
    U_441 = []
    V_441 = []
    W_441 = []
    for i in range(4):
        for j in range(4):
            u = [0] * 16
            u[i * 4 + j] = 1
            v = [0] * 4
            v[j] = 1
            w = [0] * 4
            w[i] = 1
            U_441.append(u)
            V_441.append(v)
            W_441.append(w)
            
    # 3. Combine into M_{4,4,5}
    # U: (16, 49) and (16, 16) -> (16, 65)
    # V: (16, 49) and (4, 16) -> (20, 65)
    # W: (16, 49) and (4, 16) -> (20, 65)
    
    rank_total = len(U_444) + len(U_441)
    
    U = [ [0] * rank_total for _ in range(16) ]
    V = [ [0] * rank_total for _ in range(20) ]
    W = [ [0] * rank_total for _ in range(20) ]
    
    # Fill U
    for col in range(len(U_444)):
        for row in range(16):
            U[row][col] = U_444[col][row]
    for col in range(len(U_441)):
        for row in range(16):
            U[row][len(U_444) + col] = U_441[col][row]
            
    # Fill V
    # V_444 is 16x49. Pad to 20x49.
    for col in range(len(V_444)):
        for row in range(16):
            V[row][col] = V_444[col][row]
    # V_441 is 4x16. Pad to 20x16 (top 16 zeros).
    for col in range(len(V_441)):
        for row in range(4):
            V[16 + row][len(V_444) + col] = V_441[col][row]
            
    # Fill W
    # W_444 is 16x49. Pad to 20x49.
    for col in range(len(W_444)):
        for row in range(16):
            W[row][col] = W_444[col][row]
    # W_441 is 4x16. Pad to 20x16 (top 16 zeros).
    for col in range(len(W_441)):
        for row in range(4):
            W[16 + row][len(W_444) + col] = W_441[col][row]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
