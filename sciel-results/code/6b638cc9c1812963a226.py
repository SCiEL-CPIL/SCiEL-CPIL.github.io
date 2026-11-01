"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 49 decomposition for 4x4x4 matrix multiplication
    # Using Strassen's algorithm recursively:
    # 4x4 -> 7 * (2x2)
    # 2x2 -> 7 * (1x1)
    # Total rank = 7 * 7 = 49
    
    # Define the 2x2 Strassen decomposition (Rank 7)
    # Indices for 2x2 matrices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 10, 01, 11 -> 0, 1, 2, 3 (Mapping: k*n+i with n=2. C00=0, C10=1, C01=2, C11=3)
    
    # U_2 (A coefficients), V_2 (B coefficients), W_2 (C coefficients)
    # 7 rows (rank 7), 4 cols (size 4)
    
    U_2 = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    
    V_2 = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [-1, 0, 1, 0],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    
    W_2 = [
        [1, 0, 0, 1], # M1: C00+C11 (Idx 0, 3)
        [0, 1, 0, -1],# M2: C10-C11 (Idx 1, 3) -> C10 is idx 1, C11 is idx 3
        [0, 0, 1, 1], # M3: C01+C11 (Idx 2, 3) -> C01 is idx 2, C11 is idx 3
        [1, 1, 0, 0], # M4: C00+C10 (Idx 0, 1)
        [-1, 0, 1, 0],# M5: -C00+C01 (Idx 0, 2)
        [0, 0, 0, 1], # M6: C11 (Idx 3)
        [1, 0, 0, 0]  # M7: C00 (Idx 0)
    ]
    
    # Now construct the 4x4 decomposition
    # We have 7 "outer" multiplications (k=0..6)
    # Each outer multiplication is a 2x2 matrix multiplication (l=0..6 inner rank)
    # Total rank R = 7 * 7 = 49
    
    R = 49
    U = [[0] * R for _ in range(16)]
    V = [[0] * R for _ in range(16)]
    W = [[0] * R for _ in range(16)]
    
    # Mapping of 4x4 indices to 2x2 block local indices
    # 4x4 index i (0..15) corresponds to row i//4, col i%4?
    # No, the task defines U, V, W shapes.
    # U is (n*m, rank) = (16, 49).
    # The indices 0..15 for U correspond to the entries of A.
    # Standard row-major? Or the specific convention?
    # The task says "output factor follows the official trace convention W[k*n+i,r]".
    # This implies the indices for U, V, W are flattened matrices.
    # Usually, index = row * dim + col.
    # For A (4x4): idx = r * 4 + c.
    # For B (4x4): idx = r * 4 + c.
    # For C (4x4): idx = k * 4 + i? The prompt says W[k*n+i].
    # In matrix multiplication C = AB, C[i,j] = sum_k A[i,k] B[k,j].
    # The prompt says W[k*n+i]. This is unusual.
    # Let's check the 2x2 case in the current program comments.
    # "C00 (k=0,i=0) -> 0"
    # "C10 (k=1,i=0) -> 2"
    # "C01 (k=0,i=1) -> 1"
    # "C11 (k=1,i=1) -> 3"
    # This mapping is: Index = k * n + i.
    # Here i is the row index of C? And k is the col index?
    # Standard C[i,j]. If i=row, j=col.
    # Prompt: W[k*n+i].
    # If i=row, k=col.
    # Let's assume the standard flattening for U and V is row-major: idx = r * 4 + c.
    # For W, the prompt specifies the index convention explicitly: W[k*n+i].
    # We must map the Strassen output C[i,j] to this index.
    # In Strassen, we compute C blocks.
    # C is 4x4.
    # C[i,j] where i is row, j is col.
    # The index in W array is `idx = j * 4 + i`? (If k=j, n=4, i=i).
    # Let's verify with 2x2.
    # C[0,0] -> i=0, k=0. idx = 0*2+0=0.
    # C[0,1] -> i=0, k=1. idx = 1*2+0=2.
    # C[1,0] -> i=1, k=0. idx = 0*2+1=1.
    # C[1,1] -> i=1, k=1. idx = 1*2+1=3.
    # This matches the current program's W_2 indices:
    # W_2[0] (M1) -> C00, C11.
    # If C00 is idx 0 and C11 is idx 3.
    # My W_2 above:
    # M1: [1, 0, 0, 1] -> Indices 0 and 3. Correct.
    # M2: [0, 1, 0, -1] -> Indices 1 and 3.
    # C10 is i=1, k=0 -> idx 1.
    # C11 is i=1, k=1 -> idx 3.
    # M2 formula: C10 - C11. So coeff 1 at idx 1, -1 at idx 3. Correct.
    
    # So the mapping for W index is: `idx = col * 4 + row`.
    # For U and V, standard row-major `idx = row * 4 + col`.
    
    # Block mappings
    # A blocks (U):
    # Block (br, bc) where br, bc in {0,1}
    # A[br*2+i, bc*2+j]
    # Global U index: (br*2+i)*4 + (bc*2+j)
    
    # B blocks (V):
    # B[br*2+i, bc*2+j]
    # Global V index: (br*2+i)*4 + (bc*2+j)
    
    # C blocks (W):
    # C[br*2+i, bc*2+j]
    # Global W index: (bc*2+j)*4 + (br*2+i)  # k=col, i=row
    
    # Outer Strassen Coefficients (for 4x4 blocks)
    # These are the same as U_2, V_2, W_2 but applied to blocks.
    # We iterate over the 7 outer multiplications (k_outer)
    # For each k_outer, we iterate over the 7 inner multiplications (l_inner)
    
    col = 0
    for k_outer in range(7):
        for l_inner in range(7):
            # Determine which A blocks are involved in U_2[l_inner]
            # U_2[l_inner] is a vector of 4 coeffs for the 4 A-blocks of the k_outer-th product?
            # No.
            # The structure is:
            # M_k_outer = (Sum_b A_coeff[k_outer][b] * A_block_b) * (Sum_c B_coeff[k_outer][c] * B_block_c)
            # Then M_k_outer is a 2x2 multiplication.
            # We decompose M_k_outer using the inner Strassen (U_2, V_2, W_2).
            # The "A" input to the inner Strassen is the linear combination of A blocks.
            # So, for a specific inner multiplication l_inner:
            # It picks a linear combination of the 4 entries of the "A_matrix" (which is 2x2).
            # The "A_matrix" entries are themselves linear combinations of the 4x4 A blocks.
            
            # Let's trace one scalar multiplication in the rank 49 decomposition.
            # It corresponds to a pair (k_outer, l_inner).
            # The U vector for this scalar multiplication is formed by:
            # 1. Taking the k_outer-th row of the Outer U coefficients (which are the same as U_2, let's call them U_outer).
            #    U_outer[k_outer] is a vector of 4 coeffs [c00, c01, c10, c11] for the 4 A-blocks.
            # 2. For each A-block b (0..3), we take the l_inner-th row of U_2 (inner U).
            #    U_2[l_inner] is a vector of 4 coeffs for the 4 entries of the 2x2 block.
            # 3. We combine them.
            
            # Wait, the U_2, V_2, W_2 defined above are for 2x2 *scalar* entries?
            # No, U_2 is for 2x2 matrix multiplication where the "entries" are scalars.
            # Here, the "entries" are 2x2 blocks.
            # So, the "A" in the inner Strassen is a 2x2 matrix of blocks.
            # The inner Strassen multiplies these blocks.
            # The result is a 2x2 matrix of blocks.
            # Then the outer Strassen combines these results.
            
            # So, for a fixed k_outer:
            # The "A" matrix for the inner multiplication is:
            # A_inner = [ A_outer[0]*Block00, A_outer[1]*Block01, A_outer[2]*Block10, A_outer[3]*Block11 ]
            # where A_outer is U_outer[k_outer].
            
            # Now, the inner Strassen computes products of entries of A_inner.
            # The l_inner-th multiplication in the inner Strassen takes a linear combination of the 4 entries of A_inner.
            # Let u_inner = U_2[l_inner].
            # The scalar factor for Block b is: A_outer[k_outer][b] * u_inner[local_idx].
            # This factor applies to all 4 entries of Block b.
            
            # So, for U (which is 16x49):
            # For column col:
            # For each global A index (16 of them):
            #   Identify which block b (0..3) and which local entry (0..3) it belongs to.
            #   U[global_idx][col] += A_outer[k_outer][b] * U_2[l_inner][local_idx]
            
            # Similarly for V.
            
            # For W:
            # The result of the inner multiplication is a 2x2 matrix of blocks.
            # Let W_inner = W_2[l_inner].
            # This W_inner gives coefficients for the 4 entries of the resulting 2x2 block matrix.
            # But wait. The inner multiplication produces a 2x2 matrix C_inner.
            # C_inner[local_row, local_col] = Sum ...
            # The W_2 vector tells us how to combine the 7 inner products to get C_inner entries.
            # So, for a fixed l_inner, W_2[l_inner] has 4 non-zero entries (or sparse) corresponding to the 4 entries of C_inner.
            # Each entry of C_inner is a 2x2 block of the final C.
            # So, C_inner[local_r, local_c] corresponds to the Block (br, bc) of the final C?
            # No.
            # The inner multiplication multiplies two 2x2 matrices of blocks.
            # The result is a 2x2 matrix of blocks.
            # Let's call the blocks of the result R_{br, bc}.
            # The inner Strassen computes R_{br, bc}.
            # W_2[l_inner] gives the coefficient for R_{local_idx}.
            # local_idx maps to (local_r, local_c) in the 2x2 result matrix.
            # This result matrix is then used in the outer Strassen.
            # The outer Strassen combines the 7 results (from k_outer=0..6) to get the final C blocks.
            # Let W_outer = W_2[k_outer] (using the same coefficients for the outer combination).
            # The final C block (br, bc) is:
            # C_block[br, bc] = Sum_{k_outer} W_outer[k_outer][block_idx] * R_{block_idx}^{(k_outer)}
            
            # This is getting complex to trace manually. Let's use the tensor product structure.
            # The rank 49 decomposition is the Kronecker product of the rank 7 decompositions?
            # No, it's a specific composition.
            
            # Let's stick to the explicit loop.
            
            # Precompute block mappings
            # A_block b (0..3) corresponds to:
            # b=0: rows 0,1; cols 0,1
            # b=1: rows 0,1; cols 2,3
            # b=2: rows 2,3; cols 0,1
            # b=3: rows 2,3; cols 2,3
            
            # Local indices in block (0..3):
            # 0: (0,0)
            # 1: (0,1)
            # 2: (1,0)
            # 3: (1,1)
            
            # Global U index for block b, local l:
            # row = (b // 2) * 2 + (l // 2)
            # col = (b % 2) * 2 + (l % 2)
            # U_idx = row * 4 + col
            
            # Global V index for block b, local l:
            # Same as U.
            
            # Global W index for block b, local l:
            # The result block b is at (br, bc) in the 2x2 block grid.
            # br = b // 2, bc = b % 2.
            # The local entry l is at (lr, lc) within the block.
            # Global row = br * 2 + lr
            # Global col = bc * 2 + lc
            # W_idx = Global_col * 4 + Global_row  # Convention: k*n+i -> col*4+row
            
            U_outer = U_2 # The outer coefficients are the same as inner? Yes, Strassen is self-similar.
            V_outer = V_2
            W_outer = W_2
            
            for k_outer in range(7):
                for l_inner in range(7):
                    # U
                    for b in range(4):
                        coeff_outer = U_outer[k_outer][b]
                        if coeff_outer == 0: continue
                        for l in range(4):
                            coeff_inner = U_2[l_inner][l]
                            if coeff_inner == 0: continue
                            val = coeff_outer * coeff_inner
                            if val != 0:
                                br = b // 2
                                bc = b % 2
                                lr = l // 2
                                lc = l % 2
                                row = br * 2 + lr
                                col_idx = bc * 2 + lc
                                U_idx = row * 4 + col_idx
                                U[U_idx][col] += val
                                
                    # V
                    for b in range(4):
                        coeff_outer = V_outer[k_outer][b]
                        if coeff_outer == 0: continue
                        for l in range(4):
                            coeff_inner = V_2[l_inner][l]
                            if coeff_inner == 0: continue
                            val = coeff_outer * coeff_inner
                            if val != 0:
                                br = b // 2
                                bc = b % 2
                                lr = l // 2
                                lc = l % 2
                                row = br * 2 + lr
                                col_idx = bc * 2 + lc
                                V_idx = row * 4 + col_idx
                                V[V_idx][col] += val
                                
                    # W
                    # W_2[l_inner] gives coefficients for the 4 entries of the 2x2 result block matrix.
                    # Let's say W_2[l_inner][local_res_idx] is the coeff.
                    # local_res_idx maps to (lr, lc) in the 2x2 result.
                    # This result is a block.
                    # The outer Strassen then combines these blocks.
                    # W_outer[k_outer][b_res] is the coeff for the block b_res.
                    # So for each block b_res (0..3) in the result:
                    #   coeff_outer = W_outer[k_outer][b_res]
                    #   For each local entry l_res (0..3) in that block:
                    #       coeff_inner = W_2[l_inner][l_res]
                    #       val = coeff_outer * coeff_inner
                    #       Map to global W index.
                    
                    for b_res in range(4):
                        coeff_outer = W_outer[k_outer][b_res]
                        if coeff_outer == 0: continue
                        for l_res in range(4):
                            coeff_inner = W_2[l_inner][l_res]
                            if coeff_inner == 0: continue
                            val = coeff_outer * coeff_inner
                            if val != 0:
                                br = b_res // 2
                                bc = b_res % 2
                                lr = l_res // 2
                                lc = l_res % 2
                                row = br * 2 + lr
                                col_idx = bc * 2 + lc
                                # W index: col_idx * 4 + row
                                W_idx = col_idx * 4 + row
                                W[W_idx][col] += val
                                
                    col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
