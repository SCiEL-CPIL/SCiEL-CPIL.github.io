"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    
    We partition 4x4 matrices into 2x2 blocks of 2x2 submatrices, apply Strassen
    at the block level (7 block multiplications), and apply Strassen again at the
    element level within each block (7 scalar multiplications per block).
    Total rank = 7 * 7 = 49.
    
    Interface:
    U: shape (16, 49) - U[i*m+j, r] for A[i,j]
    V: shape (16, 49) - V[j*p+k, r] for B[j,k]  
    W: shape (16, 49) - W[k*n+i, r] for C[i,k]
    """
    n, m, p = DIMENSIONS  # all 4
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Element index conventions:
    # A[i,j] -> U row i*4+j
    # B[j,k] -> V row j*4+k
    # C[i,k] -> W row k*4+i
    
    # Strassen 2x2 decomposition terms.
    # For X=[[a,b],[c,d]] * Y=[[e,f],[g,h]] = Z:
    # t1 = (a+d)(e+h) -> Z[0,0]+=1, Z[1,1]+=1
    # t2 = (c+d)e     -> Z[1,0]+=1, Z[1,1]+=-1
    # t3 = a(f-h)     -> Z[0,1]+=1, Z[1,1]+=1
    # t4 = d(g-e)     -> Z[0,0]+=1, Z[1,0]+=1
    # t5 = (a+b)h     -> Z[0,0]+=-1, Z[0,1]+=1
    # t6 = (c-a)(e+f) -> Z[1,1]+=1
    # t7 = (b-d)(g+h) -> Z[0,0]+=1
    
    # Each term defined as: (x_terms, y_terms, z_terms)
    # x_terms: list of (position, coeff) where position is (row,col) in 2x2
    # y_terms: list of (position, coeff)
    # z_terms: list of (position, coeff)
    
    strassen_terms = [
        # t1: (a+d)(e+h)
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        # t2: (c+d)e
        ([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1),((1,1),-1)]),
        # t3: a(f-h)
        ([((0,0),1)], [((0,1),1),((1,1),-1)], [((0,1),1),((1,1),1)]),
        # t4: d(g-e)
        ([((1,1),1)], [((1,0),1),((0,0),-1)], [((0,0),1),((1,0),1)]),
        # t5: (a+b)h
        ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),-1),((0,1),1)]),
        # t6: (c-a)(e+f)
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((1,1),1)]),
        # t7: (b-d)(g+h)
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((0,0),1)]),
    ]
    
    # Define the 7 block-level Strassen terms.
    # Each block term b has:
    # X_b: 2x2 matrix where each element is a linear combo of A elements
    # Y_b: 2x2 matrix where each element is a linear combo of B elements
    # W_b: mapping from block Z positions to C elements with coefficients
    
    # Block structure:
    # A11 = rows 0-1, cols 0-1; A12 = rows 0-1, cols 2-3
    # A21 = rows 2-3, cols 0-1; A22 = rows 2-3, cols 2-3
    # Similarly for B and C.
    
    # For each block term, define X and Y as 2x2 arrays of lists of (element_idx, coeff)
    # and W as dict mapping (block_row, block_col) -> dict mapping (i,k) -> coeff
    
    # M1 = (A11+A22)*(B11+B22)
    # X[0,0] = A[0,0]+A[2,2], X[0,1] = A[0,1]+A[2,3]
    # X[1,0] = A[1,0]+A[3,2], X[1,1] = A[1,1]+A[3,3]
    # Y[0,0] = B[0,0]+B[2,2], Y[0,1] = B[0,1]+B[2,3]
    # Y[1,0] = B[1,0]+B[3,2], Y[1,1] = B[1,1]+B[3,3]
    # W: C11 += 1, C22 += 1
    
    # M2 = (A21+A22)*B11
    # X[0,0] = A[2,0]+A[2,2], X[0,1] = A[2,1]+A[2,3]
    # X[1,0] = A[3,0]+A[3,2], X[1,1] = A[3,1]+A[3,3]
    # Y[0,0] = B[0,0], Y[0,1] = B[0,1]
    # Y[1,0] = B[1,0], Y[1,1] = B[1,1]
    # W: C21 += 1, C22 += -1
    
    # M3 = A11*(B12-B22)
    # X[0,0] = A[0,0], X[0,1] = A[0,1]
    # X[1,0] = A[1,0], X[1,1] = A[1,1]
    # Y[0,0] = B[0,2]-B[2,2], Y[0,1] = B[0,3]-B[2,3]
    # Y[1,0] = B[1,2]-B[3,2], Y[1,1] = B[1,3]-B[3,3]
    # W: C12 += 1, C22 += 1
    
    # M4 = A22*(B21-B11)
    # X[0,0] = A[2,2], X[0,1] = A[2,3]
    # X[1,0] = A[3,2], X[1,1] = A[3,3]
    # Y[0,0] = B[2,0]-B[0,0], Y[0,1] = B[2,1]-B[0,1]
    # Y[1,0] = B[3,0]-B[1,0], Y[1,1] = B[3,1]-B[1,1]
    # W: C11 += 1, C21 += 1
    
    # M5 = (A11+A12)*B22
    # X[0,0] = A[0,0]+A[0,2], X[0,1] = A[0,1]+A[0,3]
    # X[1,0] = A[1,0]+A[1,2], X[1,1] = A[1,1]+A[1,3]
    # Y[0,0] = B[2,2], Y[0,1] = B[2,3]
    # Y[1,0] = B[3,2], Y[1,1] = B[3,3]
    # W: C11 += -1, C12 += 1
    
    # M6 = (A21-A11)*(B11+B12)
    # X[0,0] = A[2,0]-A[0,0], X[0,1] = A[2,1]-A[0,1]
    # X[1,0] = A[3,0]-A[1,0], X[1,1] = A[3,1]-A[1,1]
    # Y[0,0] = B[0,0]+B[0,2], Y[0,1] = B[0,1]+B[0,3]
    # Y[1,0] = B[1,0]+B[1,2], Y[1,1] = B[1,1]+B[1,3]
    # W: C22 += 1
    
    # M7 = (A12-A22)*(B21+B22)
    # X[0,0] = A[0,2]-A[2,2], X[0,1] = A[0,3]-A[2,3]
    # X[1,0] = A[1,2]-A[3,2], X[1,1] = A[1,3]-A[3,3]
    # Y[0,0] = B[2,0]+B[2,2], Y[0,1] = B[2,1]+B[2,3]
    # Y[1,0] = B[3,0]+B[3,2], Y[1,1] = B[3,1]+B[3,3]
    # W: C11 += 1
    
    # Helper function to add a rank-1 term to the decomposition
    def add_term(u_row, u_coeff, v_row, v_coeff, w_row, w_coeff, col):
        if u_coeff != 0:
            U[u_row][col] += u_coeff
        if v_coeff != 0:
            V[v_row][col] += v_coeff
        if w_coeff != 0:
            W[w_row][col] += w_coeff
    
    # For each block term, apply the 7 inner Strassen terms.
    # Each inner term involves products of linear combinations of A and B elements.
    # We need to expand these into individual rank-1 terms.
    
    # For a term like (sum_i c_i * A[i]) * (sum_j d_j * B[j]) -> sum_k e_k * C[k]
    # This expands to sum_{i,j} (c_i * d_j) * A[i] * B[j] -> sum_k e_k * C[k]
    # Each (i,j) pair gives one rank-1 term: U[A[i]] = c_i, V[B[j]] = d_j, W[C[k]] = e_k
    
    # Block term definitions as (X, Y, W_map)
    # X[r][c] = list of (a_idx, coeff)
    # Y[r][c] = list of (b_idx, coeff)
    # W_map: dict from (block_row, block_col) to list of (c_idx, coeff)
    
    def get_block_defs():
        """Return list of 7 block definitions."""
        blocks = []
        
        # M1: (A11+A22)*(B11+B22)
        X = [[[(0,1),(10,1)], [(1,1),(11,1)]],
             [[(4,1),(14,1)], [(5,1),(15,1)]]]
        Y = [[[(0,1),(10,1)], [(1,1),(11,1)]],
             [[(4,1),(14,1)], [(5,1),(15,1)]]]
        W_map = {(0,0): [(0,1),(1,1),(4,1),(5,1)],  # C11: C[0,0],C[0,1],C[1,0],C[1,1]
                 (1,1): [(10,1),(11,1),(14,1),(15,1)]}  # C22: C[2,2],C[2,3],C[3,2],C[3,3]
        blocks.append((X, Y, W_map))
        
        # M2: (A21+A22)*B11
        X = [[[(8,1),(10,1)], [(9,1),(11,1)]],
             [[(12,1),(14,1)], [(13,1),(15,1)]]]
        Y = [[[(0,1)], [(1,1)]],
             [[(4,1)], [(5,1)]]]
        W_map = {(1,0): [(8,1),(9,1),(12,1),(13,1)],  # C21: C[2,0],C[2,1],C[3,0],C[3,1]
                 (1,1): [(10,-1),(11,-1),(14,-1),(15,-1)]}  # C22: negative
        blocks.append((X, Y, W_map))
        
        # M3: A11*(B12-B22)
        X = [[[(0,1)], [(1,1)]],
             [[(4,1)], [(5,1)]]]
        Y = [[[(2,1),(10,-1)], [(3,1),(11,-1)]],
             [[(6,1),(14,-1)], [(7,1),(15,-1)]]]
        W_map = {(0,1): [(2,1),(3,1),(6,1),(7,1)],  # C12: C[0,2],C[0,3],C[1,2],C[1,3]
                 (1,1): [(10,1),(11,1),(14,1),(15,1)]}  # C22
        blocks.append((X, Y, W_map))
        
        # M4: A22*(B21-B11)
        X = [[[(10,1)], [(11,1)]],
             [[(14,1)], [(15,1)]]]
        Y = [[[(8,1),(0,-1)], [(9,1),(1,-1)]],
             [[(12,1),(4,-1)], [(13,1),(5,-1)]]]
        W_map = {(0,0): [(0,1),(1,1),(4,1),(5,1)],  # C11
                 (1,0): [(8,1),(9,1),(12,1),(13,1)]}  # C21
        blocks.append((X, Y, W_map))
        
        # M5: (A11+A12)*B22
        X = [[[(0,1),(2,1)], [(1,1),(3,1)]],
             [[(4,1),(6,1)], [(5,1),(7,1)]]]
        Y = [[[(10,1)], [(11,1)]],
             [[(14,1)], [(15,1)]]]
        W_map = {(0,0): [(0,-1),(1,-1),(4,-1),(5,-1)],  # C11: negative
                 (0,1): [(2,1),(3,1),(6,1),(7,1)]}  # C12
        blocks.append((X, Y, W_map))
        
        # M6: (A21-A11)*(B11+B12)
        X = [[[(8,1),(0,-1)], [(9,1),(1,-1)]],
             [[(12,1),(4,-1)], [(13,1),(5,-1)]]]
        Y = [[[(0,1),(2,1)], [(1,1),(3,1)]],
             [[(4,1),(6,1)], [(5,1),(7,1)]]]
        W_map = {(1,1): [(10,1),(11,1),(14,1),(15,1)]}  # C22
        blocks.append((X, Y, W_map))
        
        # M7: (A12-A22)*(B21+B22)
        X = [[[(2,1),(10,-1)], [(3,1),(11,-1)]],
             [[(6,1),(14,-1)], [(7,1),(15,-1)]]]
        Y = [[[(8,1),(10,1)], [(9,1),(11,1)]],
             [[(12,1),(14,1)], [(13,1),(15,1)]]]
        W_map = {(0,0): [(0,1),(1,1),(4,1),(5,1)]}  # C11
        blocks.append((X, Y, W_map))
        
        return blocks
    
    blocks = get_block_defs()
    
    # For each block term and each inner Strassen term, expand into rank-1 terms.
    for b_idx, (X, Y, W_map) in enumerate(blocks):
        for s_idx, (x_terms, y_terms, z_terms) in enumerate(strassen_terms):
            col = b_idx * 7 + s_idx
            
            # x_terms: list of (pos, coeff) where pos is (row, col) in 2x2 X
            # y_terms: list of (pos, coeff) where pos is (row, col) in 2x2 Y
            # z_terms: list of (pos, coeff) where pos is (row, col) in 2x2 Z
            
            # For each combination of x element and y element, we get a product.
            # The product contributes to the z elements with the given coefficients.
            
            # We need to expand: (sum_x c_x * A[x_idx]) * (sum_y d_y * B[y_idx])
            # = sum_{x,y} (c_x * d_y) * A[x_idx] * B[y_idx]
            # Each such product contributes to C elements with coefficients from z_terms.
            
            for (x_pos, x_coeff) in x_terms:
                x_row, x_col = x_pos
                x_elems = X[x_row][x_col]  # list of (a_idx, coeff)
                
                for (y_pos, y_coeff) in y_terms:
                    y_row, y_col = y_pos
                    y_elems = Y[y_row][y_col]  # list of (b_idx, coeff)
                    
                    for (a_idx, a_coeff) in x_elems:
                        for (b_idx_elem, b_coeff) in y_elems:
                            # This is a rank-1 term: A[a_idx] * B[b_idx_elem]
                            # with U coeff = x_coeff * a_coeff
                            # and V coeff = y_coeff * b_coeff
                            u_coeff = x_coeff * a_coeff
                            v_coeff = y_coeff * b_coeff
                            
                            # Now distribute to C elements based on z_terms
                            for (z_pos, z_coeff) in z_terms:
                                z_row, z_col = z_pos
                                # Look up which C elements this z position maps to
                                if (z_row, z_col) in W_map:
                                    for (c_idx, c_coeff) in W_map[(z_row, z_col)]:
                                        w_coeff = z_coeff * c_coeff
                                        # Add this rank-1 term
                                        # U row for A[a_idx] is a_idx
                                        # V row for B[b_idx_elem] is b_idx_elem
                                        # W row for C[c_idx] is c_idx
                                        U[a_idx][col] += u_coeff
                                        V[b_idx_elem][col] += v_coeff
                                        W[c_idx][col] += w_coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
