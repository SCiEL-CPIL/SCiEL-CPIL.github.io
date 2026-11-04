import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 65 decomposition
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 4x4x4 multiplication using Block Strassen (Rank 49)
    # A is 4x4, B is 4x4 (cols 0-3)
    # Block indices:
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    
    # Strassen terms for 2x2 block multiplication
    # Each term is a 2x2x2 multiplication
    # We define the 7 Strassen steps
    # Step 1: (A00+A11)(B00+B11)
    # Step 2: (A10+A11)B00
    # Step 3: A00(B01-B11)
    # Step 4: A11(B10-B00)
    # Step 5: (A00+A01)B11
    # Step 6: (A10-A00)(B00+B01)
    # Step 7: (A01-A11)(B10+B11)
    
    # For each step, we perform a 2x2x2 multiplication with rank 7.
    # We need the U, V, W contributions for the 2x2x2 case.
    
    def get_strassen_2x2x2():
        # Returns lists of (u_vec, v_vec, w_vec) for the 7 terms
        # u_vec: length 4 (indices 0..3 for 2x2 A)
        # v_vec: length 4 (indices 0..3 for 2x2 B)
        # w_vec: length 4 (indices 0..3 for 2x2 C)
        
        terms = []
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # A indices: 00(0), 01(1), 10(2), 11(3)
        # B indices: 00(0), 01(1), 10(2), 11(3)
        # C indices: 00(0), 01(1), 10(2), 11(3)
        
        # M1
        u = [1, 0, 0, 1]
        v = [1, 0, 0, 1]
        w = [1, 0, 0, 1] # C00+=1, C11+=1
        terms.append((u, v, w))
        
        # M2
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        w = [0, 0, 1, -1] # C10+=1, C11-=1
        terms.append((u, v, w))
        
        # M3
        u = [1, 0, 0, 0]
        v = [0, 1, 0, -1]
        w = [0, 1, 0, 1] # C01+=1, C11+=1
        terms.append((u, v, w))
        
        # M4
        u = [0, 0, 0, 1]
        v = [0, 0, 1, -1]
        w = [1, 0, 1, 0] # C00+=1, C10+=1
        terms.append((u, v, w))
        
        # M5
        u = [1, 1, 0, 0]
        v = [0, 0, 0, 1]
        w = [-1, 1, 0, 0] # C00-=1, C01+=1
        terms.append((u, v, w))
        
        # M6
        u = [-1, 0, 1, 0]
        v = [1, 1, 0, 0]
        w = [0, 0, 0, 1] # C11+=1
        terms.append((u, v, w))
        
        # M7
        u = [0, 1, 0, -1]
        v = [0, 0, 1, 1]
        w = [1, 0, 0, 0] # C00+=1
        terms.append((u, v, w))
        
        return terms

    strassen_terms = get_strassen_2x2x2()
    
    # Map block indices to global indices
    # A block (bi, bj) where bi, bj in {0,1}
    # Global A index: (bi*2 + i) * 4 + (bj*2 + j)
    # B block (bj, bk) where bj, bk in {0,1}
    # Global B index: (bj*2 + j) * 5 + (bk*2 + k)
    # C block (bi, bk) where bi, bk in {0,1}
    # Global C index: (bk*2 + k) * 4 + (bi*2 + i)
    
    # We iterate over the 7 Strassen steps
    # Each step involves specific A and B blocks
    # Step 1: A blocks 00, 11; B blocks 00, 11
    # Step 2: A blocks 10, 11; B block 00
    # Step 3: A block 00; B blocks 01, 11
    # Step 4: A block 11; B blocks 10, 00
    # Step 5: A blocks 00, 01; B block 11
    # Step 6: A blocks 10, 00; B blocks 00, 01
    # Step 7: A blocks 01, 11; B blocks 10, 11
    
    # For each step, we have a 2x2x2 multiplication of the blocks.
    # The "A" of the 2x2x2 is formed by linear combinations of the global A blocks.
    # The "B" of the 2x2x2 is formed by linear combinations of the global B blocks.
    # The "C" of the 2x2x2 contributes to the global C blocks.
    
    # We will expand each of the 7 steps into 7 sub-steps (from strassen_terms).
    # Total 49 sub-steps.
    
    step_defs = [
        # (a_blocks, a_coeffs, b_blocks, b_coeffs, c_blocks, c_coeffs)
        # a_blocks: list of (bi, bj)
        # a_coeffs: list of coeffs
        # b_blocks: list of (bj, bk)
        # b_coeffs: list of coeffs
        # c_blocks: list of (bi, bk)
        # c_coeffs: list of coeffs
        
        # Step 1
        ([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1]),
        # Step 2
        ([(1,0), (1,1)], [1, 1], [(0,0)], [1], [(1,0), (1,1)], [1, -1]),
        # Step 3
        ([(0,0)], [1], [(0,1), (1,1)], [1, -1], [(0,1), (1,1)], [1, 1]),
        # Step 4
        ([(1,1)], [1], [(1,0), (0,0)], [1, -1], [(0,0), (1,0)], [1, 1]),
        # Step 5
        ([(0,0), (0,1)], [1, 1], [(1,1)], [1], [(0,0), (0,1)], [-1, 1]),
        # Step 6
        ([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], [(1,1)], [1]),
        # Step 7
        ([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1], [(0,0)], [1]),
    ]
    
    for step_idx, (a_blocks, a_coeffs, b_blocks, b_coeffs, c_blocks, c_coeffs) in enumerate(step_defs):
        # For each of the 7 terms in the 2x2x2 decomposition
        for t_idx, (u_vec, v_vec, w_vec) in enumerate(strassen_terms):
            # u_vec has 4 entries corresponding to the 2x2 A matrix of the step
            # v_vec has 4 entries corresponding to the 2x2 B matrix of the step
            # w_vec has 4 entries corresponding to the 2x2 C matrix of the step
            
            # We need to map the local 2x2 indices to the global blocks
            # Local A index l_a (0..3) -> Block (la_row, la_col)
            # But the step definition already maps the local 2x2 A to specific global blocks.
            # Wait, the step definition says "A blocks 00, 11" with coeffs 1, 1.
            # This forms the local 2x2 A matrix for this step.
            # The local 2x2 A matrix entries are:
            # A_local[0,0] = sum a_coeffs[i] * A_global[a_blocks[i][0], a_blocks[i][1]]
            # This is not quite right.
            
            # Let's re-evaluate.
            # In Strassen, we compute M = A_step * B_step.
            # A_step is a 2x2 matrix where each entry is a linear combination of global A blocks.
            # B_step is a 2x2 matrix where each entry is a linear combination of global B blocks.
            # C_step is a 2x2 matrix where each entry is a linear combination of global C blocks.
            
            # The 2x2x2 decomposition (strassen_terms) operates on the entries of A_step, B_step, C_step.
            # Let A_step[i,j] be the local entry.
            # A_step[i,j] = sum_{k} coeff_k * A_global[block_k]
            # Where block_k is determined by the step definition.
            
            # Actually, the step definition lists the blocks involved.
            # For Step 1: A_step[0,0] = A00, A_step[1,1] = A11, others 0?
            # No, in Strassen:
            # M1 = (A00+A11)(B00+B11).
            # The "A" part is (A00+A11). This is a scalar (in the block sense) or a matrix?
            # In block Strassen, A00, A11 are 2x2 matrices.
            # So A_step is a 2x2 matrix of 2x2 blocks?
            # No, Strassen works on 2x2 matrices of scalars.
            # Here, our "scalars" are 2x2 matrices.
            # So we are doing 2x2x2 multiplication where the "scalars" are 2x2 matrices.
            # This is equivalent to doing 2x2x2 multiplication on the block level.
            
            # The rank of 2x2x2 matrix multiplication (where entries are scalars) is 7.
            # Here, the "scalars" are 2x2 matrices.
            # So we need to decompose the 2x2x2 block multiplication.
            # This is exactly what the "strassen_terms" do, but we need to expand the "scalars" (2x2 matrices) into their components.
            
            # Let's refine the mapping.
            # For a given step (e.g., Step 1):
            # A_local[0,0] = A_global[0,0] (block 00)
            # A_local[1,1] = A_global[1,1] (block 11)
            # A_local[0,1] = 0
            # A_local[1,0] = 0
            # Wait, M1 = (A00+A11)(B00+B11).
            # This is not a 2x2 matrix multiplication. It's a product of two sums.
            # Strassen's algorithm computes 7 products M1..M7.
            # Each M is a product of two "matrices" which are sums of blocks.
            # In the standard 2x2x2 Strassen, the inputs are 2x2 matrices.
            # Here, the inputs to the 7 products are 2x2 matrices (the blocks).
            # So M1 = (A00+A11) * (B00+B11).
            # This is a 2x2 * 2x2 multiplication.
            # So for M1, we need to perform a full 2x2x2 multiplication on the blocks.
            
            # So, for each of the 7 Strassen steps, we perform a 2x2x2 multiplication on the 2x2 blocks.
            # The 2x2x2 multiplication has rank 7.
            # So total rank is 7 * 7 = 49.
            
            # How to map?
            # For Step 1:
            # A_mat = [[A00, 0], [0, A11]]? No.
            # The formula is M1 = (A00+A11)(B00+B11).
            # This can be seen as a 2x2x2 multiplication where:
            # A_in = [[A00, 0], [0, A11]]? No, that would give A00*B00 + A11*B11.
            # We want A00*B00 + A00*B11 + A11*B00 + A11*B11.
            # This is the product of (A00+A11) and (B00+B11).
            # Let X = A00+A11, Y = B00+B11.
            # M1 = X * Y.
            # X is a 2x2 matrix. Y is a 2x2 matrix.
            # So we just need to decompose X * Y.
            # X depends on A00, A11.
            # Y depends on B00, B11.
            
            # So for Step 1, we define:
            # A_local[0,0] = A00, A_local[0,1] = 0, A_local[1,0] = 0, A_local[1,1] = A11?
            # No, X = A00+A11.
            # So X[0,0] = A00[0,0] + A11[0,0], etc.
            # This means the local 2x2 matrix X is the sum of the two blocks.
            
            # So, for each step, we define the local 2x2 matrices X and Y in terms of the global blocks.
            # Then we use the 2x2x2 Strassen decomposition to compute Z = X * Y.
            # Then we map Z back to the global C blocks.
            
            # Let's implement this mapping.
            
            # Step 1: X = A00 + A11, Y = B00 + B11. Z = X*Y.
            # Z contributes to C00 and C11?
            # In the 4x4 result, M1 is added to C00 and C11.
            # C00 is the top-left 2x2 block. C11 is the bottom-right 2x2 block.
            # So Z[0,0] adds to C00, Z[0,1] adds to C01? No.
            # The product X*Y is a 2x2 matrix.
            # Let's call the result R.
            # R = X*Y.
            # In the 4x4 Strassen, M1 is added to C00 and C11.
            # This implies that the "scalar" M1 is actually the whole 2x2 matrix R.
            # And C00 += R, C11 += R.
            
            # So for Step 1:
            # A_blocks: 00, 11. Coeffs: 1, 1.
            # B_blocks: 00, 11. Coeffs: 1, 1.
            # C_blocks: 00, 11. Coeffs: 1, 1.
            
            # We need to form the local 2x2 matrices X and Y.
            # X[i,j] = sum coeff * A_block[i,j]?
            # No, the blocks are 2x2.
            # X is a 2x2 matrix.
            # X[0,0] = A00[0,0] + A11[0,0]
            # X[0,1] = A00[0,1] + A11[0,1]
            # X[1,0] = A00[1,0] + A11[1,0]
            # X[1,1] = A00[1,1] + A11[1,1]
            
            # So X is formed by summing the corresponding entries of the blocks.
            
            # We will construct the U, V, W vectors for this sub-problem.
            
            # For the current sub-term (t_idx in strassen_terms):
            # u_vec, v_vec, w_vec are for the 2x2x2 scalar multiplication.
            # We need to expand these into the global indices.
            
            # u_vec has 4 entries: u00, u01, u10, u11.
            # These multiply the entries of X.
            # X[i,j] = sum_{k} c_k * A_global[block_k][i,j]
            # So u_vec[i,j] * X[i,j] = u_vec[i,j] * sum_{k} c_k * A_global[block_k][i,j]
            # = sum_{k} (u_vec[i,j] * c_k) * A_global[block_k][i,j]
            
            # So for each global block_k involved in this step, we add to U:
            # U[global_index(block_k, i, j)] += u_vec[i,j] * c_k
            
            # Similarly for V and W.
            
            # Let's do this.
            
            # Get the definitions for this step
            # a_blocks, a_coeffs, b_blocks, b_coeffs, c_blocks, c_coeffs
            
            # We need to know which global blocks correspond to the local 2x2 matrix entries.
            # In Step 1, X is formed from A00 and A11.
            # X[i,j] = A00[i,j] + A11[i,j].
            # So for local index (i,j), the contributing global blocks are 00 and 11.
            
            # Generalizing:
            # For a step, the local matrix X is a sum of global blocks.
            # X[i,j] = sum_{k} a_coeffs[k] * A_global[a_blocks[k]][i,j]
            
            # So for each local index (i,j) in 0..1:
            #   For each k in range(len(a_blocks)):
            #       global_block = a_blocks[k]
            #       coeff = a_coeffs[k]
            #       global_A_index = (global_block[0]*2 + i)*4 + (global_block[1]*2 + j)
            #       U[global_A_index] += u_vec[i*2+j] * coeff
            
            # Similarly for V and W.
            
            for i in range(2):
                for j in range(2):
                    local_idx = i*2 + j
                    u_val = u_vec[local_idx]
                    v_val = v_vec[local_idx]
                    w_val = w_vec[local_idx]
                    
                    # Update U
                    for k in range(len(a_blocks)):
                        bi, bj = a_blocks[k]
                        coeff = a_coeffs[k]
                        global_i = bi*2 + i
                        global_j = bj*2 + j
                        global_A_idx = global_i*4 + global_j
                        U[global_A_idx][col] += u_val * coeff
                        
                    # Update V
                    for k in range(len(b_blocks)):
                        bj, bk = b_blocks[k]
                        coeff = b_coeffs[k]
                        global_j = bj*2 + j
                        global_k = bk*2 + k
                        global_B_idx = global_j*5 + global_k
                        V[global_B_idx][col] += v_val * coeff
                        
                    # Update W
                    for k in range(len(c_blocks)):
                        bi, bk = c_blocks[k]
                        coeff = c_coeffs[k]
                        global_i = bi*2 + i
                        global_k = bk*2 + k
                        global_C_idx = global_k*4 + global_i
                        W[global_C_idx][col] += w_val * coeff
            
            col += 1

    # Part 2: 4x4x1 multiplication (5th column) using Schoolbook (Rank 16)
    # C[i, 4] = sum_{j=0}^3 A[i, j] * B[j, 4]
    for i in range(4):
        for j in range(4):
            global_A_idx = i*4 + j
            global_B_idx = j*5 + 4
            global_C_idx = 4*4 + i # k=4, so 4*4 + i
            
            U[global_A_idx][col] = 1
            V[global_B_idx][col] = 1
            W[global_C_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
