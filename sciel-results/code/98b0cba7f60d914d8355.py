"""Rewritten program for 4x4 matrix multiplication rank-minimal decomposition."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # We'll use the recursive Strassen decomposition.
    # 4x4 = 2x2 blocks of 2x2 matrices.
    # Each 2x2 block multiplication uses Strassen (7 terms).
    # The 2x2 block-level Strassen uses 7 terms.
    # Total rank = 7 * 7 = 49.
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Strassen 2x2 decomposition terms.
    # For X = [[a,b],[c,d]] * Y = [[e,f],[g,h]]:
    # t1 = (a+d)(e+h)
    # t2 = (c+d)e
    # t3 = a(f-h)
    # t4 = d(g-e)
    # t5 = (a+b)h
    # t6 = (c-a)(e+f)
    # t7 = (b-d)(g+h)
    # Result:
    # p11 = t1 + t4 - t5 + t7
    # p12 = t3 + t5
    # p21 = t2 + t4
    # p22 = t1 - t2 + t3 + t6
    
    # Each term as (u_coeff, v_coeff, w_coeff) for [a,b,c,d] x [e,f,g,h] -> [p11,p12,p21,p22]
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # t1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # t2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # t3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),   # t4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # t5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # t6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # t7
    ]
    
    # Block-level Strassen for 4x4:
    # Partition into 2x2 blocks of 2x2 submatrices.
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # Similarly for B and C.
    
    # Block-level Strassen terms:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each block term, we need to define:
    # - The 2x2 X matrix (from A side) as linear combinations of A elements
    # - The 2x2 Y matrix (from B side) as linear combinations of B elements
    # - Which C blocks this term contributes to, with what coefficients
    
    # Element indexing:
    # A[i,j] -> U row i*4+j
    # B[j,k] -> V row j*4+k
    # C[i,k] -> W row k*4+i
    
    # Define the 7 block terms.
    # Each term has:
    # - X: 2x2 matrix where each entry is a list of (A_elem_idx, coeff)
    # - Y: 2x2 matrix where each entry is a list of (B_elem_idx, coeff)
    # - W_map: dict from (block_row, block_col) to coeff for C blocks
    
    # A element indices:
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    # A[2,0]=8, A[2,1]=9, A[2,2]=10, A[2,3]=11
    # A[3,0]=12, A[3,1]=13, A[3,2]=14, A[3,3]=15
    
    # B element indices:
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3
    # B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    # B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    # B[3,0]=12, B[3,1]=13, B[3,2]=14, B[3,3]=15
    
    # C element indices (W row = k*4+i for C[i,k]):
    # C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3
    # C[1,0]=4, C[1,1]=5, C[1,2]=6, C[1,3]=7
    # C[2,0]=8, C[2,1]=9, C[2,2]=10, C[2,3]=11
    # C[3,0]=12, C[3,1]=13, C[3,2]=14, C[3,3]=15
    
    # Block term definitions:
    blocks = []
    
    # M1: X = A11+A22, Y = B11+B22, W: C11+=1, C22+=1
    X = [[[(0,1),(10,1)], [(1,1),(11,1)]],
         [[(4,1),(14,1)], [(5,1),(15,1)]]]
    Y = [[[(0,1),(10,1)], [(1,1),(11,1)]],
         [[(4,1),(14,1)], [(5,1),(15,1)]]]
    W_map = {(0,0):1, (1,1):1}
    blocks.append({'X': X, 'Y': Y, 'W': W_map})
    
    # M2: X = A21+A22, Y = B11, W: C21+=1, C22+=-1
    X = [[[(8,1),(10,1)], [(9,1),(11,1)]],
         [[(12,1),(14,1)], [(13,1),(15,1)]]]
    Y = [[[(0,1)], [(1,1)]],
         [[(4,1)], [(5,1)]]]
    W_map = {(1,0):1, (1,1):-1}
    blocks.append({'X': X, 'Y': Y, 'W': W_map})
    
    # M3: X = A11, Y = B12-B22, W: C12+=1, C22+=1
    X = [[[(0,1)], [(1,1)]],
         [[(4,1)], [(5,1)]]]
    Y = [[[(2,1),(10,-1)], [(3,1),(11,-1)]],
         [[(6,1),(14,-1)], [(7,1),(15,-1)]]]
    W_map = {(0,1):1, (1,1):1}
    blocks.append({'X': X, 'Y': Y, 'W': W_map})
    
    # M4: X = A22, Y = B21-B11, W: C11+=1, C21+=1
    X = [[[(10,1)], [(11,1)]],
         [[(14,1)], [(15,1)]]]
    Y = [[[(8,1),(0,-1)], [(9,1),(1,-1)]],
         [[(12,1),(4,-1)], [(13,1),(5,-1)]]]
    W_map = {(0,0):1, (1,0):1}
    blocks.append({'X': X, 'Y': Y, 'W': W_map})
    
    # M5: X = A11+A12, Y = B22, W: C11+=-1, C12+=1
    X = [[[(0,1),(2,1)], [(1,1),(3,1)]],
         [[(4,1),(6,1)], [(5,1),(7,1)]]]
    Y = [[[(10,1)], [(11,1)]],
         [[(14,1)], [(15,1)]]]
    W_map = {(0,0):-1, (0,1):1}
    blocks.append({'X': X, 'Y': Y, 'W': W_map})
    
    # M6: X = A21-A11, Y = B11+B12, W: C22+=1
    X = [[[(8,1),(0,-1)], [(9,1),(1,-1)]],
         [[(12,1),(4,-1)], [(13,1),(5,-1)]]]
    Y = [[[(0,1),(2,1)], [(1,1),(3,1)]],
         [[(4,1),(6,1)], [(5,1),(7,1)]]]
    W_map = {(1,1):1}
    blocks.append({'X': X, 'Y': Y, 'W': W_map})
    
    # M7: X = A12-A22, Y = B21+B22, W: C11+=1
    X = [[[(2,1),(10,-1)], [(3,1),(11,-1)]],
         [[(6,1),(14,-1)], [(7,1),(15,-1)]]]
    Y = [[[(8,1),(10,1)], [(9,1),(11,1)]],
         [[(12,1),(14,1)], [(13,1),(15,1)]]]
    W_map = {(0,0):1}
    blocks.append({'X': X, 'Y': Y, 'W': W_map})
    
    # Now for each block term, apply 2x2 Strassen to decompose X*Y.
    # The 2x2 Strassen terms decompose X*Y into 7 scalar products.
    # Each scalar product involves:
    # - A linear combination of X elements (from the u_coeff of the Strassen term)
    # - A linear combination of Y elements (from the v_coeff of the Strassen term)
    # - Contributions to the result Z elements (from the w_coeff of the Strassen term)
    
    # For each Strassen term s with (u_coeff, v_coeff, w_coeff):
    # x_scalar = u_coeff[0]*X[0][0] + u_coeff[1]*X[0][1] + u_coeff[2]*X[1][0] + u_coeff[3]*X[1][1]
    # y_scalar = v_coeff[0]*Y[0][0] + v_coeff[1]*Y[0][1] + v_coeff[2]*Y[1][0] + v_coeff[3]*Y[1][1]
    # Z[0][0] += w_coeff[0] * x_scalar * y_scalar
    # Z[0][1] += w_coeff[1] * x_scalar * y_scalar
    # Z[1][0] += w_coeff[2] * x_scalar * y_scalar
    # Z[1][1] += w_coeff[3] * x_scalar * y_scalar
    
    # In the tensor decomposition, each such term is a rank-1 term where:
    # - U has the coefficients for x_scalar (linear combo of A elements)
    # - V has the coefficients for y_scalar (linear combo of B elements)
    # - W has the coefficients for the Z elements (linear combo of C elements)
    
    # For each block b and Strassen term s:
    # col = b * 7 + s
    # Build U side: for each A element, sum up the coefficients from x_scalar
    # Build V side: for each B element, sum up the coefficients from y_scalar
    # Build W side: for each C element, sum up the coefficients from Z elements
    
    for b_idx, block in enumerate(blocks):
        X = block['X']
        Y = block['Y']
        W_map = block['W']
        
        for s_idx, (u_coeff, v_coeff, w_coeff) in enumerate(strassen_2x2):
            col = b_idx * 7 + s_idx
            
            # Build x_side: dict from A_elem_idx -> total_coeff
            x_side = {}
            x_positions = [(0,0), (0,1), (1,0), (1,1)]
            for pos_idx, (r, c) in enumerate(x_positions):
                uc = u_coeff[pos_idx]
                if uc == 0:
                    continue
                for elem_idx, elem_coeff in X[r][c]:
                    x_side[elem_idx] = x_side.get(elem_idx, 0) + uc * elem_coeff
            
            # Build y_side: dict from B_elem_idx -> total_coeff
            y_side = {}
            y_positions = [(0,0), (0,1), (1,0), (1,1)]
            for pos_idx, (r, c) in enumerate(y_positions):
                vc = v_coeff[pos_idx]
                if vc == 0:
                    continue
                for elem_idx, elem_coeff in Y[r][c]:
                    y_side[elem_idx] = y_side.get(elem_idx, 0) + vc * elem_coeff
            
            # Build w_side: dict from C_elem_idx -> total_coeff
            # Z[0][0] contributes to C elements in block (0,0) if W_map has (0,0)
            # Z[0][1] contributes to C elements in block (0,1) if W_map has (0,1)
            # Z[1][0] contributes to C elements in block (1,0) if W_map has (1,0)
            # Z[1][1] contributes to C elements in block (1,1) if W_map has (1,1)
            w_side = {}
            z_positions = [(0,0), (0,1), (1,0), (1,1)]
            for pos_idx, (br, bc) in enumerate(z_positions):
                wc = w_coeff[pos_idx]
                if wc == 0:
                    continue
                if (br, bc) not in W_map:
                    continue
                block_coeff = W_map[(br, bc)]
                # Z[br][bc] is a 2x2 matrix, so it contributes to 4 C elements
                # C[br*2+dr, bc*2+dc] for dr, dc in 0,1
                for dr in range(2):
                    for dc in range(2):
                        i = br * 2 + dr  # row of C
                        k = bc * 2 + dc  # col of C
                        c_elem_idx = k * 4 + i  # W row index
                        w_side[c_elem_idx] = w_side.get(c_elem_idx, 0) + wc * block_coeff
            
            # Now populate U, V, W for this column
            for elem_idx, coeff in x_side.items():
                if coeff != 0:
                    U[elem_idx][col] += coeff
            
            for elem_idx, coeff in y_side.items():
                if coeff != 0:
                    V[elem_idx][col] += coeff
            
            for elem_idx, coeff in w_side.items():
                if coeff != 0:
                    W[elem_idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
