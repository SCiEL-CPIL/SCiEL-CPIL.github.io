import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 part + 16 for 4x4x1 part = 65
    rank = 65
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # A is 4x4, B is 4x4 (cols 0-3)
    # Blocks:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # C11: rows 0-1, cols 0-1
    # C12: rows 0-1, cols 2-3
    # C21: rows 2-3, cols 0-1
    # C22: rows 2-3, cols 2-3
    
    # Strassen Steps (u_coeffs for A blocks, v_coeffs for B blocks)
    # Each step is a list of (block_idx, coeff) for U, V, and W (result block, coeff)
    # A blocks: 0=A11, 1=A12, 2=A21, 3=A22
    # B blocks: 0=B11, 1=B12, 2=B21, 3=B22
    # C blocks: 0=C11, 1=C12, 2=C21, 3=C22
    
    # Inner 2x2 Strassen (Rank 7)
    # u_inner, v_inner, w_inner (coeffs for 2x2 blocks)
    # 2x2 indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    inner_u = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [0, 1, -1, 0],# M6: b-a
        [0, 0, 1, -1] # M7: c-d
    ]
    inner_v = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [0, 0, 1, -1],# M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 0, 1, 0], # M6: e+g
        [0, 1, 0, 1]  # M7: f+h
    ]
    inner_w = [
        # M1 contributes to C11 (+1), C22 (+1)
        [1, 0, 0, 1], 
        # M2 contributes to C11 (-1), C21 (+1)
        [-1, 0, 1, 0],
        # M3 contributes to C12 (+1), C22 (-1)
        [0, 1, 0, -1],
        # M4 contributes to C11 (+1), C21 (+1)
        [1, 0, 1, 0],
        # M5 contributes to C12 (+1), C22 (+1)
        [0, 1, 0, 1],
        # M6 contributes to C11 (+1)
        [1, 0, 0, 0],
        # M7 contributes to C22 (-1)
        [0, 0, 0, -1]
    ]
    
    # Strassen Outer Steps
    # Each step defines which A-blocks and B-blocks are summed/subtracted
    # And which C-blocks receive the result
    # Format: (u_block_coeffs, v_block_coeffs, w_block_coeffs)
    # u_block_coeffs: dict {block_idx: coeff}
    
    outer_steps = [
        # M1 = (A11+A22)(B11+B22)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # Result goes to C11 and C22? No, the inner w handles that.
        # Wait, the inner w is for the 2x2 result of the block multiplication.
        # The block multiplication result is a 2x2 matrix.
        # So the inner w maps to the 4 sub-blocks of the result block?
        # No. The "block" in Strassen is a 2x2 matrix.
        # So (A11+A22) is a 2x2 matrix. (B11+B22) is a 2x2 matrix.
        # Their product is a 2x2 matrix.
        # This 2x2 matrix IS one of the C-blocks?
        # No. C is 4x4.
        # C11 is 2x2.
        # So M1 is a 2x2 matrix.
        # M1 contributes to C11 and C22?
        # Let's check Strassen formulas again.
        # C11 = M1 + M4 - M2 + M6
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M5 - M7
        
        # So M1 (which is a 2x2 matrix) adds to C11 and C22.
        # So for Step 1, the result of the 2x2 multiplication is added to C-block 0 and C-block 3.
        # So w_outer for Step 1 is {0: 1, 3: 1}.
        
        # M2 = (A21+A22)B11
        # u: A21, A22. v: B11.
        # Result adds to C11 (-1) and C21 (+1).
        
        # M3 = A11(B12-B22)
        # u: A11. v: B12, -B22.
        # Result adds to C12 (+1) and C22 (-1).
        
        # M4 = A22(B21-B11)
        # u: A22. v: B21, -B11.
        # Result adds to C11 (+1) and C21 (+1).
        
        # M5 = (A11+A12)B22
        # u: A11, A12. v: B22.
        # Result adds to C12 (+1) and C22 (+1).
        
        # M6 = (A12-A11)(B11+B21)
        # u: A12, -A11. v: B11, B21.
        # Result adds to C11 (+1).
        
        # M7 = (A21-A22)(B12+B22)
        # u: A21, -A22. v: B12, B22.
        # Result adds to C22 (-1).
    ]
    
    outer_steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {0: -1, 2: 1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 3: 1}),
        ({1: 1, 0: -1}, {0: 1, 2: 1}, {0: 1}),
        ({2: 1, 3: -1}, {1: 1, 3: 1}, {3: -1})
    ]
    
    for step_idx in range(7):
        u_blocks, v_blocks, w_blocks = outer_steps[step_idx]
        
        for t in range(7):
            # Get inner vectors
            u_in = inner_u[t]
            v_in = inner_v[t]
            w_in = inner_w[t]
            
            # Construct global U, V, W for this term
            # U index: i*m + j
            # V index: j*p + k
            # W index: k*n + i
            
            # u_in is for 2x2 A-block.
            # u_in[0] is A00, u_in[1] is A01, u_in[2] is A10, u_in[3] is A11.
            
            # We need to map this to the 4x4 A.
            # The A-blocks in the step are specified in u_blocks.
            # e.g. if u_blocks has {0: 1, 3: 1}, it means A11 and A22.
            # The "2x2" structure of the block multiplication implies we are multiplying two 2x2 matrices.
            # The 2x2 matrix A_block has entries A00, A01, A10, A11.
            # These correspond to the 4 sub-entries of the 2x2 block.
            
            # Wait, the Strassen algorithm operates on the blocks.
            # A11 is a 2x2 matrix.
            # So "A11" in the formula refers to the whole 2x2 block.
            # The inner Strassen decomposition is for the 2x2 matrix multiplication.
            # So the u_in vector applies to the entries of the 2x2 block.
            
            # So, for a given outer step:
            # We have a linear combination of A-blocks.
            # Let A_comb = sum( coeff * A_block ).
            # A_comb is a 2x2 matrix.
            # We then multiply A_comb by B_comb (also 2x2).
            # The inner Strassen decomposes this 2x2x2 product.
            
            # So, for each entry (a,b) in the 2x2 u_in vector:
            # It corresponds to entry (a,b) of the 2x2 block A_comb.
            # Which global entries of A contribute to A_comb[a,b]?
            # If A_comb = 1*A11 + 1*A22.
            # A11 is at global (0,0), (0,1), (1,0), (1,1).
            # A22 is at global (2,0), (2,1), (3,0), (3,1).
            # So A_comb[0,0] = A11[0,0] + A22[0,0] = A[0,0] + A[2,0].
            
            # So we need to map the 2x2 block indices to global indices.
            
            # Block 0 (A11): rows 0,1; cols 0,1.
            # Block 1 (A12): rows 0,1; cols 2,3.
            # Block 2 (A21): rows 2,3; cols 0,1.
            # Block 3 (A22): rows 2,3; cols 2,3.
            
            block_rows = {0: [0,1], 1: [0,1], 2: [2,3], 3: [2,3]}
            block_cols = {0: [0,1], 1: [2,3], 2: [0,1], 3: [2,3]}
            
            for b_idx, coeff in u_blocks.items():
                r1, r2 = block_rows[b_idx]
                c1, c2 = block_cols[b_idx]
                
                # u_in[0] is entry (0,0) of the block
                # u_in[1] is entry (0,1)
                # u_in[2] is entry (1,0)
                # u_in[3] is entry (1,1)
                
                # Map to global i, j
                # (0,0) -> i=r1, j=c1
                # (0,1) -> i=r1, j=c2
                # (1,0) -> i=r2, j=c1
                # (1,1) -> i=r2, j=c2
                
                indices = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
                for k, (i, j) in enumerate(indices):
                    U[i * m + j, col] += coeff * u_in[k]
            
            # Similarly for V
            # B blocks:
            # Block 0 (B11): rows 0,1; cols 0,1.
            # Block 1 (B12): rows 0,1; cols 2,3.
            # Block 2 (B21): rows 2,3; cols 0,1.
            # Block 3 (B22): rows 2,3; cols 2,3.
            
            block_rows_B = {0: [0,1], 1: [0,1], 2: [2,3], 3: [2,3]}
            block_cols_B = {0: [0,1], 1: [2,3], 2: [0,1], 3: [2,3]}
            
            for b_idx, coeff in v_blocks.items():
                r1, r2 = block_rows_B[b_idx]
                c1, c2 = block_cols_B[b_idx]
                
                indices = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
                for k, (j, k_idx) in enumerate(indices):
                    # Note: B index is j*p + k_idx
                    V[j * p + k_idx, col] += coeff * v_in[k]
            
            # Similarly for W
            # C blocks (for the 4x4 part, cols 0-3)
            # Block 0 (C11): rows 0,1; cols 0,1.
            # Block 1 (C12): rows 0,1; cols 2,3.
            # Block 2 (C21): rows 2,3; cols 0,1.
            # Block 3 (C22): rows 2,3; cols 2,3.
            
            block_rows_C = {0: [0,1], 1: [0,1], 2: [2,3], 3: [2,3]}
            block_cols_C = {0: [0,1], 1: [2,3], 2: [0,1], 3: [2,3]}
            
            for b_idx, coeff in w_blocks.items():
                r1, r2 = block_rows_C[b_idx]
                c1, c2 = block_cols_C[b_idx]
                
                indices = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
                for k, (k_idx, i) in enumerate(indices):
                    # Note: W index is k*n + i
                    W[k_idx * n + i, col] += coeff * w_in[k]
            
            col += 1
    
    # Part 2: 4x4x1 using Schoolbook (Rank 16)
    # A is 4x4, B is 4x1 (col 4)
    # C is 4x1 (col 4)
    
    for i in range(4):
        for j in range(4):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1 # k=4
            col += 1
    
    return {"U": U, "V": V, "W": W}
