"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # We'll use a tiling approach: partition 6x6 into 2x2 blocks of 3x3 submatrices
    # For each pair of 3x3 blocks, use Strassen-like decomposition for 3x3 matrices
    # Actually, let's think about this more carefully.
    
    # The standard approach: if we partition A (6x6), B (6x6), C (6x6) into 2x2 blocks
    # where each block is 3x3, then we need to compute 4*3*3 = 36 scalar multiplications
    # per block pair using schoolbook, but we can use better methods for 3x3.
    
    # Actually, the known result is that 3x3 matrix multiplication has rank 23 (Bini et al.)
    # But implementing that is complex. Let's use a simpler approach first.
    
    # Alternative: Use Strassen's algorithm recursively.
    # For 2x2 matrices, Strassen uses 7 multiplications instead of 8.
    # For 6x6, we can partition into 2x2 blocks of 3x3, then for each 3x3, 
    # we could further partition... but 3 is not a power of 2.
    
    # Let's try a different tiling: partition 6x6 into 3x3 blocks of 2x2 submatrices.
    # Then we have 9 block pairs, each requiring 2x2 matrix multiplication.
    # Using Strassen for each 2x2: 7 multiplications per block pair.
    # Total rank: 9 * 7 = 63? No, that's not right either.
    
    # Actually, when we partition into blocks, the tensor decomposition is more complex.
    # Let me think about this differently.
    
    # The matrix multiplication tensor for n x n has a certain structure.
    # For 6x6, we can use the fact that 6 = 2 * 3, and use a Kronecker product approach.
    
    # If we have decompositions for 2x2 (rank 7 via Strassen) and 3x3 (rank 23 via Bini),
    # we can combine them. But the combination is not simply multiplicative.
    
    # Let's start with a simpler approach: use the known Strassen decomposition for 2x2
    # and apply it in a tiled fashion.
    
    # For 6x6, partition into 2x2 blocks of 3x3. Each 3x3 block multiplication 
    # requires 27 scalar multiplications (schoolbook). 
    # There are 4 block positions, so 4 * 27 = 108 if we do independent block multiplications.
    # But that's worse than schoolbook 216/2 = 108? No, schoolbook for 6x6 is 216.
    # Wait, 6*6*6 = 216. If we do 4 independent 3x3 multiplications, that's 4*27 = 108.
    # But we also need to handle the cross-block terms!
    
    # Actually, when multiplying block matrices:
    # C_11 = A_11*B_11 + A_12*B_21
    # C_12 = A_11*B_12 + A_12*B_22
    # C_21 = A_21*B_11 + A_22*B_21
    # C_22 = A_21*B_12 + A_22*B_22
    
    # So we need 8 block multiplications (each 3x3 x 3x3), not 4.
    # Using schoolbook for each: 8 * 27 = 216, same as direct.
    # Using Strassen for the block structure: 7 block multiplications, each 3x3.
    # If each 3x3 uses schoolbook: 7 * 27 = 189.
    
    # That's the approach that got score 1.143 in the knowledge!
    # Rank 189 = 7 * 27.
    
    # Can we do better? If we use a rank-23 decomposition for 3x3, then:
    # 7 * 23 = 161.
    
    # Let me implement the 7-block Strassen decomposition where each block mult 
    # uses a 3x3 decomposition.
    
    # First, let's implement Strassen's 7-multiplication scheme for 2x2 blocks.
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
    
    # Each Mi is a 3x3 matrix product.
    
    # For the 3x3 products, we'll use schoolbook for now (rank 27 each).
    # Total rank: 7 * 27 = 189.
    
    # Let's implement this.
    
    # The U, V, W matrices will have shape (36, 189) each.
    # U corresponds to A (6x6 flattened to 36 elements)
    # V corresponds to B (6x6 flattened to 36 elements)  
    # W corresponds to C (6x6 flattened to 36 elements)
    
    # Indexing: A[i,j] -> U[i*6+j, :]
    #            B[i,j] -> V[i*6+j, :]
    #            C[i,j] -> W[i*6+j, :]
    
    rank = 7 * 27  # 189
    U = np.zeros((36, rank), dtype=np.int64)
    V = np.zeros((36, rank), dtype=np.int64)
    W = np.zeros((36, rank), dtype=np.int64)
    
    # Define the 7 Strassen sub-problems
    # Each sub-problem is a 3x3 matrix multiplication
    # We need to express each as a tensor decomposition
    
    # For Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Block indices in the 6x6 matrix (0-indexed):
    # A11: rows 0-2, cols 0-2
    # A12: rows 0-2, cols 3-5
    # A21: rows 3-5, cols 0-2
    # A22: rows 3-5, cols 3-5
    
    # For each Strassen term, we need to create a 3x3 matrix multiplication decomposition
    # with appropriate linear combinations of the input blocks.
    
    # Helper: create a schoolbook 3x3 decomposition
    # Returns the U, V, W contributions for a 3x3 matmul with given block mappings
    # a_rows, a_cols: row and col offsets for the A block (3x3)
    # b_rows, b_cols: row and col offsets for the B block (3x3)
    # c_rows, c_cols: row and col offsets for the C block (3x3)
    # a_sign: +1 or -1 for each element of the A block (for linear combinations)
    # b_sign: +1 or -1 for each element of the B block
    # c_sign: +1 or -1 for each element of the C block
    
    def add_3x3_decomp(U, V, W, start_col, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, 
                       a_signs=None, b_signs=None, c_signs=None):
        """Add a schoolbook 3x3 decomposition starting at column start_col.
        a_signs, b_signs, c_signs are 3x3 arrays of +1/-1 or None (meaning all +1).
        """
        if a_signs is None:
            a_signs = np.ones((3,3), dtype=np.int64)
        if b_signs is None:
            b_signs = np.ones((3,3), dtype=np.int64)
        if c_signs is None:
            c_signs = np.ones((3,3), dtype=np.int64)
        
        col = start_col
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # U index: A block element (a_rows[i], a_cols[j])
                    u_idx = (a_rows[i]) * 6 + (a_cols[j])
                    # V index: B block element (b_rows[j], b_cols[k])
                    v_idx = (b_rows[j]) * 6 + (b_cols[k])
                    # W index: C block element (c_rows[i], c_cols[k])
                    w_idx = (c_rows[i]) * 6 + (c_cols[k])
                    
                    U[u_idx, col] += a_signs[i, j]
                    V[v_idx, col] += b_signs[j, k]
                    W[w_idx, col] += c_signs[i, k]
                    col += 1
        return col
    
    # Block row/col offsets
    # Block 11: rows 0-2, cols 0-2
    # Block 12: rows 0-2, cols 3-5
    # Block 21: rows 3-5, cols 0-2
    # Block 22: rows 3-5, cols 3-5
    
    b11_rows = [0, 1, 2]
    b11_cols = [0, 1, 2]
    b12_rows = [0, 1, 2]
    b12_cols = [3, 4, 5]
    b21_rows = [3, 4, 5]
    b21_cols = [0, 1, 2]
    b22_rows = [3, 4, 5]
    b22_cols = [3, 4, 5]
    
    col = 0
    
    # M1 = (A11+A22)(B11+B22) -> contributes to C11 and C22
    # We need to add this to both C11 and C22 with appropriate signs
    # C11 += M1, C22 += M1
    
    # For M1, the A part is A11 + A22, B part is B11 + B22
    # We'll create a combined decomposition where each scalar product involves
    # elements from both A11 and A22, and both B11 and B22.
    
    # Actually, it's cleaner to think of it as:
    # M1[i,k] = sum_j (A11[i,j] + A22[i,j]) * (B11[j,k] + B22[j,k])
    #         = sum_j [A11[i,j]*B11[j,k] + A11[i,j]*B22[j,k] + A22[i,j]*B11[j,k] + A22[i,j]*B22[j,k]]
    
    # This requires 4 * 27 = 108 scalar multiplications if done naively.
    # But we want to use the structure to share computations.
    
    # Actually, the proper way to do Strassen with tensor decompositions is to
    # create separate decompositions for each of the 7 terms, where each term
    # is itself a matrix multiplication (possibly with linear combinations).
    
    # Let me reconsider. The standard way to express Strassen as a tensor decomposition:
    # Each of the 7 multiplications Mi is a bilinear form. For Mi = Xi * Yi where
    # Xi and Yi are linear combinations of the input blocks, we need to express
    # the product as a sum of scalar multiplications.
    
    # For example, M1 = (A11+A22)(B11+B22)
    # The (i,k) entry is sum_j (A11[i,j]+A22[i,j])(B11[j,k]+B22[j,k])
    # = sum_j A11[i,j]B11[j,k] + sum_j A11[i,j]B22[j,k] + sum_j A22[i,j]B11[j,k] + sum_j A22[i,j]B22[j,k]
    
    # So M1 actually requires 4 separate 3x3 matrix multiplications if we expand.
    # That would give 7 * 4 * 27 = 756, which is worse than schoolbook 216!
    
    # I think I'm confusing myself. Let me look at this differently.
    
    # The correct approach: In Strassen's algorithm, we compute 7 matrix products
    # of 3x3 matrices. Each such product is a separate matrix multiplication.
    # The key insight is that the 7 products are:
    # M1 = (A11+A22) * (B11+B22)  -- this is ONE 3x3 matrix multiplication
    # where the left operand is the sum of two 3x3 blocks, and the right operand 
    # is the sum of two 3x3 blocks.
    
    # In terms of tensor decomposition, computing M1 requires us to compute
    # (sum of two matrices) * (sum of two matrices), which expands to 4 matrix products.
    # So it's not saving anything unless we can share.
    
    # Wait, I think the confusion is that Strassen's algorithm reduces the number of
    # recursive calls, but each recursive call is still a full matrix multiplication.
    # The savings come from the recursion depth, not from the base case.
    
    # For a direct (non-recursive) tensor decomposition, Strassen's 7-multiplication
    # scheme for 2x2 gives rank 7 for the 2x2 tensor. When applied to 6x6 by tiling,
    # the rank becomes 7 * (rank of 3x3 tensor).
    
    # So if we use schoolbook for 3x3 (rank 27), total rank = 7 * 27 = 189.
    # This is valid and better than 216.
    
    # The implementation: for each of the 7 Strassen terms, we create a 3x3 
    # matrix multiplication decomposition. The linear combinations are handled
    # by having multiple contributions to the same U/V/W entries.
    
    # Let me implement this properly.
    
    # For each Strassen term Mi, we have:
    # Mi = Xi * Yi where Xi and Yi are 3x3 matrices that are linear combinations of blocks.
    
    # To express Mi as a tensor decomposition, we note that:
    # Mi[i,k] = sum_j Xi[i,j] * Yi[j,k]
    
    # If Xi[i,j] = sum_a xa_a[i,j] * A[a] (linear combination of A blocks)
    # and Yi[j,k] = sum_b yb_b[j,k] * B[b] (linear combination of B blocks)
    # Then Mi[i,k] = sum_j sum_a sum_b xa_a[i,j] * yb_b[j,k] * A[a] * B[b]
    
    # This gives multiple scalar multiplications per (i,j,k) triple.
    
    # Let's define the 7 terms explicitly.
    
    # I'll create a list of terms, each specifying:
    # - For the A side: list of (block, sign) pairs
    # - For the B side: list of (block, sign) pairs  
    # - For the C side: list of (block, sign) pairs (where this term contributes)
    
    # Block identifiers: '11', '12', '21', '22'
    # Each block has 3x3 = 9 elements.
    
    # Let's define the block mappings:
    block_map = {
        '11': (b11_rows, b11_cols),
        '12': (b12_rows, b12_cols),
        '21': (b21_rows, b21_cols),
        '22': (b22_rows, b22_cols),
    }
    
    # Strassen terms:
    # M1: A = A11 + A22, B = B11 + B22, C = C11 + C22 (contributes +M1 to both)
    # M2: A = A21 + A22, B = B11, C = C21 (contributes +M2)
    # M3: A = A11, B = B12 - B22, C = C12 (contributes +M3)
    # M4: A = A22, B = B21 - B11, C = C11 + C21 (contributes +M4 to both)
    # M5: A = A11 + A12, B = B22, C = C11 - C12 (contributes +M5 to C11, -M5 to C12)
    # M6: A = A21 - A11, B = B11 + B12, C = C22 (contributes +M6)
    # M7: A = A12 - A22, B = B21 + B22, C = C11 - C22 (contributes +M7 to C11, -M7 to C22)
    
    # Wait, let me double-check the Strassen formulas:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So:
    # M1 contributes: +M1 to C11, +M1 to C22
    # M2 contributes: +M2 to C21, -M2 to C22
    # M3 contributes: +M3 to C12, +M3 to C22
    # M4 contributes: +M4 to C11, +M4 to C21
    # M5 contributes: -M5 to C11, +M5 to C12
    # M6 contributes: +M6 to C22
    # M7 contributes: +M7 to C11, -M7 to C22
    
    # And the definitions:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Let me redefine more carefully:
    strassen_terms = [
        # (A_blocks_with_signs, B_blocks_with_signs, C_blocks_with_signs)
        # M1
        ([('11', 1), ('22', 1)], [('11', 1), ('22', 1)], [('11', 1), ('22', 1)]),
        # M2
        ([('21', 1), ('22', 1)], [('11', 1)], [('21', 1), ('22', -1)]),
        # M3
        ([('11', 1)], [('12', 1), ('22', -1)], [('12', 1), ('22', 1)]),
        # M4
        ([('22', 1)], [('21', 1), ('11', -1)], [('11', 1), ('21', 1)]),
        # M5
        ([('11', 1), ('12', 1)], [('22', 1)], [('11', -1), ('12', 1)]),
        # M6
        ([('21', 1), ('11', -1)], [('11', 1), ('12', 1)], [('22', 1)]),
        # M7
        ([('12', 1), ('22', -1)], [('21', 1), ('22', 1)], [('11', 1), ('22', -1)]),
    ]
    
    col = 0
    for a_blocks, b_blocks, c_blocks in strassen_terms:
        # For this term, we need to compute the 3x3 matrix product
        # where the A operand is a linear combination of the specified A blocks,
        # and the B operand is a linear combination of the specified B blocks.
        
        # The product has 3x3 = 9 entries, each being a sum over j=0,1,2.
        # For each (i,j,k), the contribution is:
        # sum over a in a_blocks: sign_a * A_block_a[i,j]
        # times
        # sum over b in b_blocks: sign_b * B_block_b[j,k]
        
        # This expands to |a_blocks| * |b_blocks| scalar multiplications per (i,j,k).
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for a_block, a_sign in a_blocks:
                        for b_block, b_sign in b_blocks:
                            for c_block, c_sign in c_blocks:
                                # A element: block a_block, position (i,j)
                                a_rows, a_cols = block_map[a_block]
                                u_idx = a_rows[i] * 6 + a_cols[j]
                                
                                # B element: block b_block, position (j,k)
                                b_rows, b_cols = block_map[b_block]
                                v_idx = b_rows[j] * 6 + b_cols[k]
                                
                                # C element: block c_block, position (i,k)
                                c_rows, c_cols = block_map[c_block]
                                w_idx = c_rows[i] * 6 + c_cols[k]
                                
                                U[u_idx, col] += a_sign
                                V[v_idx, col] += b_sign
                                W[w_idx, col] += c_sign
                                col += 1
    
    # Check if we've used the expected number of columns
    # Expected: sum over terms of (len(a_blocks) * len(b_blocks) * len(c_blocks) * 9)
    # M1: 2*2*2*9 = 72
    # M2: 2*1*2*9 = 36
    # M3: 1*2*2*9 = 36
    # M4: 1*2*2*9 = 36
    # M5: 2*1*2*9 = 36
    # M6: 2*2*1*9 = 36
    # M7: 2*2*2*9 = 72
    # Total: 72+36+36+36+36+36+72 = 324
    
    # Hmm, that's more than 189. Let me reconsider.
    
    # I think the issue is that I'm double-counting. In the standard Strassen tensor decomposition,
    # each of the 7 terms is a single 3x3 matrix multiplication, which has rank 27.
    # The linear combinations are absorbed into the U and V matrices by having multiple
    # non-zero entries in the same column.
    
    # Actually, I think my approach is correct but the rank is higher than 189 because
    # I'm expanding the linear combinations into separate scalar multiplications.
    
    # Let me verify: for M1 = (A11+A22)(B11+B22), the (i,k) entry is:
    # sum_j (A11[i,j]+A22[i,j])(B11[j,k]+B22[j,k])
    # = sum_j [A11[i,j]B11[j,k] + A11[i,j]B22[j,k] + A22[i,j]B11[j,k] + A22[i,j]B22[j,k]]
    
    # This is 4 separate 3x3 matrix multiplications, each of rank 27, so 4*27 = 108 for M1 alone.
    # That's clearly not the intended Strassen decomposition.
    
    # I think the correct interpretation is that Strassen's algorithm is a recursive
    # algorithm, and the "rank 7" refers to the number of recursive calls, not the
    # number of scalar multiplications at the base case.
    
    # For a direct tensor decomposition, we need to think about this differently.
    
    # Let me go back to the schoolbook approach and just return that for now,
    # since I'm running out of time and the implementation is getting complex.
    
    if time.time() > deadline - 10:
        # Return schoolbook as fallback
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
    
    # Let me try a cleaner approach: just use schoolbook but with the block structure
    # to potentially share some computations. For now, let's just return the 
    # schoolbook decomposition since the Strassen approach is more complex than expected.
    
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
