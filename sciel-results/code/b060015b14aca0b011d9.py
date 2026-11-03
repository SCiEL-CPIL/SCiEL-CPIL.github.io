import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 49
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen patterns for 2x2 matrix multiplication
    # Each pattern is a tuple of (U_indices, U_values, V_indices, V_values, W_indices, W_values)
    # Indices are 0..3 for 2x2 matrices (row-major)
    strassen_patterns = [
        # P1 = (m00 + m11)(n00 + n11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # P2 = (m10 + m11)n00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # P3 = m00(n10 - n11)
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1), (3, 1)]),
        # P4 = m11(n01 - n00)
        ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (2, 1)]),
        # P5 = (m00 + m01)n11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # P6 = (m10 - m00)(n00 + n01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # P7 = (m01 - m11)(n10 + n11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # Strassen steps for 4x4 block multiplication
    # Each step k defines X_k and Y_k as 2x2 matrices of blocks
    # X_k = [[X00, X01], [X10, X11]]
    # Y_k = [[Y00, Y01], [Y10, Y11]]
    # Each Xab is a linear combination of A_blocks
    # A_blocks are indexed 0..3 (row-major)
    # A00, A01, A10, A11
    # X00, X01, X10, X11 are 2x2 matrices of coefficients for A_blocks
    
    # Step 1: M1 = (A00 + A11)(B00 + B11)
    # X1 = [[A00, 0], [0, A11]]? No.
    # X1 is a 2x2 matrix of blocks.
    # (A00 + A11) is a block. Let's call it S1.
    # S1 = A00 + A11.
    # X1 = [[S1, 0], [0, 0]]? No, X1 is the left operand.
    # The product is S1 * S2 where S2 = B00 + B11.
    # So X1 = S1 (as a 2x2 matrix of scalars? No, S1 is a 2x2 matrix of scalars).
    # Wait, A00 is a 2x2 matrix of scalars.
    # So X1 is a 2x2 matrix of scalars?
    # No, the "block" multiplication is between 2x2 matrices of blocks.
    # The result M1 is a 2x2 matrix of blocks (i.e., a 4x4 matrix of scalars).
    # But Strassen step 1 computes M1 = (A00+A11)(B00+B11).
    # A00+A11 is a 2x2 matrix of scalars.
    # B00+B11 is a 2x2 matrix of scalars.
    # So M1 is a 2x2 matrix of scalars.
    # This M1 is then used in the final combination for C.
    
    # So for each Strassen step k, we are multiplying two 2x2 matrices of scalars.
    # Let X_k and Y_k be these 2x2 matrices.
    # X_k entries are linear combinations of A's entries.
    # Y_k entries are linear combinations of B's entries.
    
    # Define the 7 Strassen steps for the block level
    # X_k and Y_k are 2x2 matrices.
    # We represent them as 4x1 vectors of coefficients for the 16 A/B entries?
    # No, X_k is 2x2. Each entry is a linear combination of 16 A entries.
    # So X_k can be represented as a 2x2x16 tensor?
    # Or simply, for each entry (a,b) of X_k, we have a vector of 16 coefficients.
    
    # Let's define the block-level Strassen steps
    # Step 1: X = [[A00+A11, 0], [0, 0]]? No.
    # The formula is M1 = (A00+A11)(B00+B11).
    # This is a product of two 2x2 matrices (which are blocks).
    # So X1 = A00+A11 (as a 2x2 matrix).
    # Y1 = B00+B11 (as a 2x2 matrix).
    
    # How to represent X1?
    # X1[0,0] = A00[0,0] + A11[0,0]
    # X1[0,1] = A00[0,1] + A11[0,1]
    # etc.
    
    # We will define a list of 7 steps.
    # Each step has X_coeffs and Y_coeffs.
    # X_coeffs is a 2x2 matrix, each element is a list of 16 coefficients.
    
    def get_block_coeffs(block_idx, sign):
        # block_idx 0..3 for A00, A01, A10, A11
        # Returns a 16-vector of coefficients
        coeffs = [0]*16
        r0 = block_idx // 2
        c0 = block_idx % 2
        for i in range(2):
            for j in range(2):
                idx = (r0*2 + i)*4 + (c0*2 + j)
                coeffs[idx] = sign
        return coeffs
    
    steps = []
    
    # Step 1: X = A00+A11, Y = B00+B11
    X1 = [
        [get_block_coeffs(0, 1), get_block_coeffs(0, 1)], # Row 0: A00+A11? No.
        # A00 is block 0. A11 is block 3.
        # X1 is a 2x2 matrix.
        # X1[0,0] = (A00+A11)[0,0]
        # X1[0,1] = (A00+A11)[0,1]
        # X1[1,0] = (A00+A11)[1,0]
        # X1[1,1] = (A00+A11)[1,1]
    ]
    # Actually, X1 is just the matrix A00+A11.
    # So X1[a,b] = (A00+A11)[a,b].
    # (A00+A11)[a,b] = A00[a,b] + A11[a,b].
    # A00[a,b] is at index a*4+b? No, A00 is a block.
    # A00 is at rows 0,1 and cols 0,1.
    # A00[a,b] is at global index (0*2+a)*4 + (0*2+b) = a*4+b.
    # A11[a,b] is at global index (1*2+a)*4 + (1*2+b) = (2+a)*4 + (2+b) = 8+4a+2+b.
    
    # Let's write a helper to get the coefficient vector for a block sum
    def block_sum_coeffs(blocks):
        # blocks is a list of (idx, sign)
        coeffs = [0]*16
        for idx, sign in blocks:
            r0 = idx // 2
            c0 = idx % 2
            for a in range(2):
                for b in range(2):
                    global_idx = (r0*2 + a)*4 + (c0*2 + b)
                    coeffs[global_idx] += sign
        return coeffs

    # Step 1: X = A00+A11, Y = B00+B11
    X1 = [
        [block_sum_coeffs([(0,1), (3,1)]), block_sum_coeffs([(0,1), (3,1)])], # This is wrong. X1 is a 2x2 matrix.
        # X1[0,0] is the (0,0) element of the matrix A00+A11.
        # The matrix A00+A11 has elements:
        # [A00[0,0]+A11[0,0], A00[0,1]+A11[0,1]]
        # [A00[1,0]+A11[1,0], A00[1,1]+A11[1,1]]
        # So X1[0,0] corresponds to A00[0,0] and A11[0,0].
        # X1[0,1] corresponds to A00[0,1] and A11[0,1].
    ]
    
    # It is easier to define X_k as a 2x2 matrix of 16-vectors.
    # X_k[a,b] is the coefficient vector for the (a,b) element of the k-th Strassen left operand.
    
    # Let's define the 7 steps explicitly.
    
    # Step 1: M1 = (A00+A11)(B00+B11)
    # X1 is the matrix A00+A11.
    # X1[0,0] = A00[0,0] + A11[0,0]
    # X1[0,1] = A00[0,1] + A11[0,1]
    # X1[1,0] = A00[1,0] + A11[1,0]
    # X1[1,1] = A00[1,1] + A11[1,1]
    
    def make_X(block_specs):
        # block_specs is a list of 4 lists, each containing (block_idx, sign)
        # Returns a 2x2 matrix of 16-vectors
        X = [[None]*2 for _ in range(2)]
        for a in range(2):
            for b in range(2):
                coeffs = [0]*16
                for idx, sign in block_specs[a*2+b]:
                    r0 = idx // 2
                    c0 = idx % 2
                    global_idx = (r0*2 + a)*4 + (c0*2 + b)
                    coeffs[global_idx] += sign
                X[a][b] = coeffs
        return X

    # Step 1
    X1 = make_X([
        [(0,1), (3,1)], # X1[0,0]
        [(0,1), (3,1)], # X1[0,1]
        [(0,1), (3,1)], # X1[1,0]
        [(0,1), (3,1)]  # X1[1,1]
    ])
    Y1 = make_X([
        [(0,1), (3,1)], # Y1[0,0]
        [(0,1), (3,1)], # Y1[0,1]
        [(0,1), (3,1)], # Y1[1,0]
        [(0,1), (3,1)]  # Y1[1,1]
    ])
    steps.append((X1, Y1))
    
    # Step 2: M2 = (A10+A11)B00
    # X2 is the matrix A10+A11.
    # X2[0,0] = A10[0,0] + A11[0,0]
    # X2[0,1] = A10[0,1] + A11[0,1]
    # X2[1,0] = A10[1,0] + A11[1,0]
    # X2[1,1] = A10[1,1] + A11[1,1]
    X2 = make_X([
        [(1,1), (3,1)],
        [(1,1), (3,1)],
        [(1,1), (3,1)],
        [(1,1), (3,1)]
    ])
    Y2 = make_X([
        [(0,1)], # Y2[0,0] = B00[0,0]
        [(0,1)], # Y2[0,1] = B00[0,1]
        [(0,1)], # Y2[1,0] = B00[1,0]
        [(0,1)]  # Y2[1,1] = B00[1,1]
    ])
    steps.append((X2, Y2))
    
    # Step 3: M3 = A00(B10-B11)
    X3 = make_X([
        [(0,1)],
        [(0,1)],
        [(0,1)],
        [(0,1)]
    ])
    Y3 = make_X([
        [(1,1), (3,-1)],
        [(1,1), (3,-1)],
        [(1,1), (3,-1)],
        [(1,1), (3,-1)]
    ])
    steps.append((X3, Y3))
    
    # Step 4: M4 = A11(B01-B00)
    X4 = make_X([
        [(3,1)],
        [(3,1)],
        [(3,1)],
        [(3,1)]
    ])
    Y4 = make_X([
        [(1,1), (0,-1)],
        [(1,1), (0,-1)],
        [(1,1), (0,-1)],
        [(1,1), (0,-1)]
    ])
    steps.append((X4, Y4))
    
    # Step 5: M5 = (A00+A01)B11
    X5 = make_X([
        [(0,1), (1,1)],
        [(0,1), (1,1)],
        [(0,1), (1,1)],
        [(0,1), (1,1)]
    ])
    Y5 = make_X([
        [(3,1)],
        [(3,1)],
        [(3,1)],
        [(3,1)]
    ])
    steps.append((X5, Y5))
    
    # Step 6: M6 = (A10-A00)(B00+B01)
    X6 = make_X([
        [(1,1), (0,-1)],
        [(1,1), (0,-1)],
        [(1,1), (0,-1)],
        [(1,1), (0,-1)]
    ])
    Y6 = make_X([
        [(0,1), (1,1)],
        [(0,1), (1,1)],
        [(0,1), (1,1)],
        [(0,1), (1,1)]
    ])
    steps.append((X6, Y6))
    
    # Step 7: M7 = (A01-A11)(B10+B11)
    X7 = make_X([
        [(1,1), (3,-1)],
        [(1,1), (3,-1)],
        [(1,1), (3,-1)],
        [(1,1), (3,-1)]
    ])
    Y7 = make_X([
        [(1,1), (3,1)],
        [(1,1), (3,1)],
        [(1,1), (3,1)],
        [(1,1), (3,1)]
    ])
    steps.append((X7, Y7))
    
    # Now, for each step k (0..6), we multiply X_k and Y_k using the 7 Strassen patterns.
    # This gives 49 terms.
    
    col = 0
    for k in range(7):
        X_k, Y_k = steps[k]
        for s in range(7):
            u_indices, v_indices, w_indices = strassen_patterns[s]
            # u_indices is a list of (index_in_2x2, value)
            # We need to map index_in_2x2 to global indices in A/B
            
            # For U:
            # The s-th Strassen term for 2x2 multiplication is:
            # U_s = sum_{a,b} u_{s,ab} X_k[ab]
            # X_k[ab] is a 16-vector of coefficients for A.
            # So U_s is a 16-vector: sum_{a,b} u_{s,ab} * X_k[ab]
            
            U_vec = [0]*16
            for idx, val in u_indices:
                a = idx // 2
                b = idx % 2
                for i in range(16):
                    U_vec[i] += val * X_k[a][b][i]
                    
            for i in range(16):
                U[i][col] = U_vec[i]
                
            # For V:
            V_vec = [0]*16
            for idx, val in v_indices:
                a = idx // 2
                b = idx % 2
                for i in range(16):
                    V_vec[i] += val * Y_k[a][b][i]
                    
            for i in range(16):
                V[i][col] = V_vec[i]
                
            # For W:
            # The s-th Strassen term contributes to M_k.
            # M_k is a 2x2 matrix of blocks.
            # M_k[ab] gets contribution w_{s,ab} from the s-th term.
            # M_k is then combined to form C.
            # C[ab] = sum_k coeff_{k,ab} M_k
            # We need to add to W the contribution of M_k[ab] to C[ab].
            
            # The final combination for C:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # So for step k, the coefficient of M_k[ab] in C[ab] is:
            # Let's define a 2x2 matrix of coefficients for each k.
            
            # We will handle this by adding to W directly.
            # W is indexed by (k_global, i_global).
            # M_k[ab] is a 2x2 matrix of scalars.
            # M_k[ab][cd] contributes to C[2a+c, 2b+d]?
            # No, C is 4x4.
            # M_k is a 2x2 matrix of 2x2 matrices.
            # So M_k[ab] is a 2x2 matrix.
            # Its element (c,d) is at global C index (2a+c, 2b+d).
            
            # The s-th Strassen term contributes w_{s,ab} to M_k[ab].
            # So it contributes w_{s,ab} * M_k[ab][cd] to C[2a+c, 2b+d].
            
            # We need to add to W[global_C_index][col] the value w_{s,ab} * coeff_{k,ab}.
            # Where coeff_{k,ab} is the coefficient of M_k[ab] in C[ab].
            
            # Let's precompute coeff_{k,ab}
            # k=0 (M1): C00(+1), C11(+1)
            # k=1 (M2): C10(+1), C11(-1)
            # k=2 (M3): C01(+1), C11(+1)
            # k=3 (M4): C00(+1), C10(+1)
            # k=4 (M5): C00(-1), C01(+1)
            # k=5 (M6): C11(+1)
            # k=6 (M7): C00(+1)
            
            # Note: C00 is block (0,0), C01 is (0,1), etc.
            # M_k[ab] is the block at (a,b) of the 2x2 block matrix M_k.
            # So M_k[0,0] is C00, M_k[0,1] is C01, etc.
            
            coeffs = {
                0: {(0,0): 1, (1,1): 1},
                1: {(1,0): 1, (1,1): -1},
                2: {(0,1): 1, (1,1): 1},
                3: {(0,0): 1, (1,0): 1},
                4: {(0,0): -1, (0,1): 1},
                5: {(1,1): 1},
                6: {(0,0): 1}
            }
            
            for idx, val in w_indices:
                a = idx // 2
                b = idx % 2
                coeff = coeffs[k].get((a,b), 0)
                if coeff == 0:
                    continue
                # M_k[ab] is a 2x2 matrix.
                # Its elements (c,d) contribute to C[2a+c, 2b+d].
                # The value is val * coeff.
                for c in range(2):
                    for d in range(2):
                        global_i = 2*a + c
                        global_j = 2*b + d
                        # W index is k*n + i? No.
                        # W is (p*n, rank).
                        # Index is (k_global, i_global).
                        # In the task, W[k*n+i, r].
                        # Here, the output index is (row, col) of C.
                        # C is 4x4.
                        # So W index is (row*4 + col).
                        w_idx = global_i * 4 + global_j
                        W[w_idx][col] += val * coeff
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
