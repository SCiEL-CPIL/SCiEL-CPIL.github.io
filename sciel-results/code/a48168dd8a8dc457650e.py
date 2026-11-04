import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank 34
    # 12 for 2x1x6
    # 11 for first 2x2x3
    # 11 for second 2x2x3
    
    rank = 34
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Part 1: A1 (2x1) * B1 (1x6)
    # A1 corresponds to A[:, 0] (indices 0, 1 in A flattened)
    # B1 corresponds to B[0, :] (indices 0..5 in B flattened)
    # Output C[:, 0:6] (indices 0..5 in C flattened? No, W index is k*n+i)
    # C is 2x6. C[i, k]. W index k*n + i.
    # A1 is 2x1. B1 is 1x6.
    # C[i, k] = A1[i, 0] * B1[0, k]
    for i in range(2):
        for k in range(6):
            U[i*3 + 0, col] = 1 # A index i*3 + 0
            V[0*6 + k, col] = 1 # B index 0*6 + k
            W[k*2 + i, col] = 1 # W index k*2 + i
            col += 1
            
    # Part 2: A2 (2x2) * B2 (2x6)
    # A2 is A[:, 1:3] (indices 1, 2, 3, 4 in A flattened)
    # B2 is B[1:3, :] (indices 6..17 in B flattened)
    # B2 is split into B21 (2x3) and B22 (2x3)
    # B21 is B[1:3, 0:3] (indices 6,7,8, 9,10,11)
    # B22 is B[1:3, 3:6] (indices 12,13,14, 15,16,17)
    
    def add_2x2x3_decomp(a_offset, b_offset, w_offset):
        # a_offset: index in U for A[0,0] of the 2x2 block
        # b_offset: index in V for B[0,0] of the 2x3 block
        # w_offset: index in W for C[0,0] of the 2x3 block
        # A is 2x2. Indices in A: 
        # A[0,0] -> a_offset
        # A[0,1] -> a_offset+1
        # A[1,0] -> a_offset+2
        # A[1,1] -> a_offset+3
        # Note: A is flattened row-major? 
        # Original A is 2x3. 
        # A2 is A[:, 1:3]. 
        # A[0,1] is index 1. A[0,2] is index 2.
        # A[1,1] is index 4. A[1,2] is index 5.
        # So a_offset for A2 is 1.
        # A2[0,0] = A[0,1] (idx 1)
        # A2[0,1] = A[0,2] (idx 2)
        # A2[1,0] = A[1,1] (idx 4)
        # A2[1,1] = A[1,2] (idx 5)
        
        # B is 2x3. Indices in B:
        # B[0,0] -> b_offset
        # B[0,1] -> b_offset+1
        # B[0,2] -> b_offset+2
        # B[1,0] -> b_offset+3
        # B[1,1] -> b_offset+4
        # B[1,2] -> b_offset+5
        
        # W is 2x3. Indices in W (k*n+i):
        # C[0,0] -> w_offset
        # C[0,1] -> w_offset+1
        # C[0,2] -> w_offset+2
        # C[1,0] -> w_offset+3
        # C[1,1] -> w_offset+4
        # C[1,2] -> w_offset+5
        
        global col
        
        # Strassen for 2x2x2 part (cols 0,1 of B)
        # M1 = A00 * (B00 + B11)
        U[a_offset, col] = 1
        V[b_offset, col] = 1
        V[b_offset+4, col] = 1
        W[w_offset, col] = 0.5 # C00 = M1 + M4 - M2 - M7? 
        # Let's use standard Strassen reconstruction
        # C00 = M1 + M4 - M2 - M7
        # C01 = M3 + M7
        # C10 = M5 + M6
        # C11 = M1 + M2 - M4 + M6
        
        # Wait, I need to fill U, V, W for each multiplication.
        # The W matrix accumulates the results.
        # So for each multiplier, I add the contribution to W.
        
        def add_mult(a_idx, v_indices, w_indices, w_coeff):
            nonlocal col
            U[a_idx, col] = 1
            for vi in v_indices:
                V[vi, col] = 1
            for wi, coeff in zip(w_indices, w_coeff):
                W[wi, col] += coeff
            col += 1
            
        # M1
        add_mult(a_offset, [b_offset, b_offset+4], [w_offset, w_offset+3], [0.5, 0.5]) # C00, C10? 
        # C00 = M1 + M4 - M2 - M7
        # C10 = M5 + M6
        # C01 = M3 + M7
        # C11 = M1 + M2 - M4 + M6
        
        # Let's map correctly:
        # M1 = A00 (B00+B11)
        # M2 = (A00+A10) B11
        # M3 = (A01+A11) B00
        # M4 = A11 (B00+B11)
        # M5 = (A00+A01) B10
        # M6 = (A10-A11) (B00+B01)
        # M7 = (A01-A10) (B10+B11)
        
        # C00 = M1 + M4 - M2 - M7
        # C01 = M3 + M7
        # C10 = M5 + M6
        # C11 = M1 + M2 - M4 + M6
        
        # I already added M1 partially? No, I used a helper.
        # Let's restart the helper logic inside the function properly.
        
        # Clear previous partial work if any? No, col is global.
        # I will just write the 11 multipliers directly.
        
        # M1
        U[a_offset, col] = 1
        V[b_offset, col] = 1
        V[b_offset+4, col] = 1
        W[w_offset, col] = 1 # C00
        W[w_offset+3, col] = 1 # C10? No, C10 is M5+M6.
        # Wait, C00 gets M1. C11 gets M1.
        W[w_offset, col] = 1
        W[w_offset+4, col] = 1
        col += 1
        
        # M2
        U[a_offset, col] = 1
        U[a_offset+2, col] = 1
        V[b_offset+4, col] = 1
        W[w_offset, col] = -1
        W[w_offset+4, col] = 1
        col += 1
        
        # M3
        U[a_offset+1, col] = 1
        U[a_offset+3, col] = 1
        V[b_offset, col] = 1
        W[w_offset+1, col] = 1
        col += 1
        
        # M4
        U[a_offset+3, col] = 1
        V[b_offset, col] = 1
        V[b_offset+4, col] = 1
        W[w_offset, col] = 1
        W[w_offset+4, col] = -1
        col += 1
        
        # M5
        U[a_offset, col] = 1
        U[a_offset+1, col] = 1
        V[b_offset+3, col] = 1
        W[w_offset+3, col] = 1
        col += 1
        
        # M6
        U[a_offset+2, col] = 1
        U[a_offset+3, col] = -1
        V[b_offset, col] = 1
        V[b_offset+1, col] = 1
        W[w_offset+3, col] = 1
        W[w_offset+4, col] = 1
        col += 1
        
        # M7
        U[a_offset+1, col] = 1
        U[a_offset+2, col] = -1
        V[b_offset+3, col] = 1
        V[b_offset+4, col] = 1
        W[w_offset, col] = -1
        W[w_offset+1, col] = 1
        col += 1
        
        # Schoolbook for col 2 (B02, B12)
        # C02 = A00 B02 + A01 B12
        # C12 = A10 B02 + A11 B12
        
        # M8 = A00 B02
        U[a_offset, col] = 1
        V[b_offset+2, col] = 1
        W[w_offset+2, col] = 1
        col += 1
        
        # M9 = A01 B12
        U[a_offset+1, col] = 1
        V[b_offset+5, col] = 1
        W[w_offset+2, col] = 1
        col += 1
        
        # M10 = A10 B02
        U[a_offset+2, col] = 1
        V[b_offset+2, col] = 1
        W[w_offset+5, col] = 1
        col += 1
        
        # M11 = A11 B12
        U[a_offset+3, col] = 1
        V[b_offset+5, col] = 1
        W[w_offset+5, col] = 1
        col += 1
        
    # Call for B21 (cols 0-2 of B2, which are cols 0-2 of B)
    # A2 offset 1
    # B21 offset 6 (B[1,0] is index 6)
    # W offset for C[:, 0:3] is 0
    add_2x2x3_decomp(1, 6, 0)
    
    # Call for B22 (cols 3-5 of B2, which are cols 3-5 of B)
    # A2 offset 1
    # B22 offset 12 (B[1,3] is index 12)
    # W offset for C[:, 3:6] is 3
    add_2x2x3_decomp(1, 12, 3)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
