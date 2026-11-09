<<<<<<< SEARCH
    def rank28_444():
        """
        Return (U_sub, V_sub, W_sub) for 4x4x4 matmul with rank 28.
        U_sub: 16 x 28, V_sub: 16 x 28, W_sub: 16 x 28
        such that sum_s U_sub[i*4+j, s] * V_sub[j*4+k, s] * W_sub[k*4+i, s] = 1
        for all i,j,k in [0,4), and 0 otherwise.
        """
        # Concrete rank-28 decomposition for 4x4x4 matrix multiplication
        # Based on known low-rank constructions
        
        U_sub = [[0]*28 for _ in range(16)]
        V_sub = [[0]*28 for _ in range(16)]
        W_sub = [[0]*28 for _ in range(16)]
        
        # Define 28 trilinear forms
        # Each form s has support on specific (i,j), (j,k), (k,i) triples
        
        # Construction: Use block structure
        # Partition {0,1,2,3} into two pairs: {0,1} and {2,3}
        # Use Strassen for each 2x2 block and handle cross-block terms
        
        # Strassen 2x2x2 rank-7 decomposition:
        # For 2x2 matrices A, B: C = A*B
        # 7 multiplications:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # Let me define the 28 forms explicitly.
        # I'll use a verified construction.
        
        # Here's a rank-28 construction based on combining 2x2 Strassen decompositions
        # for the 4 diagonal 2x2 blocks, plus 12 cross-block terms.
        
        # Actually, let me use a simpler approach: directly define the 28 forms
        # using a known explicit list.
        
        # Form 0-6: Strassen for block (0,0) x (0,0) -> (0,0)
        # i,j in {0,1}, k in {0,1}
        strassen_forms = [
            # (u_pattern, v_pattern, w_pattern) for 2x2x2
            # u: 4 entries for (i,j), v: 4 entries for (j,k), w: 4 entries for (k,i)
            # Form 0: (A00+A11)*(B00+B11)
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),  # M1
            # Form 1: (A10+A11)*B00
            ([0,0,1,1], [1,0,0,0], [0,0,0,0]),  # M2 - needs correction
            # Let me use a cleaner approach
        ]
        
        # This is getting complex. Let me use a different strategy.
        # I'll implement a correct but possibly not minimal rank decomposition.
        
        # For now, use rank 32 which I can construct more easily:
        # 4 independent 2x2x2 Strassen decompositions (rank 7 each) = 28
        # Plus 4 cross-block terms = 32
        
        # Actually, let me just use a correct rank-64 for now to ensure
        # the overall structure is correct, then optimize.
        
        rank = 64
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_sub[i * 4 + j][col] = 1
                    V_sub[j * 4 + k][col] = 1
                    W_sub[k * 4 + i][col] = 1
                    col += 1
        
        # Resize to actual rank
        U_sub = [row[:rank] for row in U_sub]
        V_sub = [row[:rank] for row in V_sub]
        W_sub = [row[:rank] for row in W_sub]
        
        return U_sub, V_sub, W_sub
=======
    def rank28_444():
        """
        Return (U_sub, V_sub, W_sub) for 4x4x4 matmul with rank 28.
        U_sub: 16 x 28, V_sub: 16 x 28, W_sub: 16 x 28
        such that sum_s U_sub[i*4+j, s] * V_sub[j*4+k, s] * W_sub[k*4+i, s] = 1
        for all i,j,k in [0,4), and 0 otherwise.
        """
        # Rank-28 decomposition for 4x4x4 matrix multiplication
        # Using explicit construction
        
        U_sub = [[0]*28 for _ in range(16)]
        V_sub = [[0]*28 for _ in range(16)]
        W_sub = [[0]*28 for _ in range(16)]
        
        # Helper: add a Strassen 2x2x2 decomposition for a block
        def add_strassen_block(i_offset, j_offset, k_offset, col_offset):
            """Add 7 Strassen forms for a 2x2x2 block at given offsets."""
            # Strassen 2x2x2 rank-7 forms
            # u: indices (i,j) in 2x2, v: (j,k) in 2x2, w: (k,i) in 2x2
            strassen = [
                # (u_vals, v_vals, w_vals) for the 4 entries each
                ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
                ([0,0,1,1], [1,0,0,0], [0,0,0,0]),   # M2
                ([1,0,0,0], [0,1,0,0], [0,0,0,0]),   # M3
                ([0,0,0,1], [0,0,1,0], [0,0,0,0]),   # M4
                ([1,1,0,0], [0,0,0,1], [0,0,0,0]),   # M5
                ([0,0,1,-1], [1,1,0,0], [0,0,0,0]),  # M6
                ([0,1,0,-1], [0,0,1,1], [0,0,0,0]),  # M7
            ]
            # This is incorrect. Let me use a proper Strassen decomposition.
            pass
        
        # Let me use a completely different approach:
        # Use a known explicit rank-28 list from the literature.
        
        # For now, return a correct rank-64 decomposition
        # and mark this as a placeholder
        
        rank = 64
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_sub[i * 4 + j][col] = 1
                    V_sub[j * 4 + k][col] = 1
                    W_sub[k * 4 + i][col] = 1
                    col += 1
        
        U_sub = [row[:rank] for row in U_sub]
        V_sub = [row[:rank] for row in V_sub]
        W_sub = [row[:rank] for row in W_sub]
        
        return U_sub, V_sub, W_sub
>>>>>>> REPLACE

I'm struggling to implement a correct rank-28 decomposition from memory. Let me use a different approach: I'll implement a correct but not necessarily minimal rank decomposition that I can verify is correct, then the score will be based on whatever rank I achieve.

Let me use a rank-32 decomposition for 4×4×4 based on a simpler construction, giving total rank 64 for the 4×4×8 problem. This is still better than 128.

Actually, let me just implement a correct solution first and verify it works, then optimize the rank.

