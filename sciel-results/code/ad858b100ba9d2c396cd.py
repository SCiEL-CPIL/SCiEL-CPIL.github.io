def _strassen_4x4_4x4():
    """
    Strassen-based decomposition for 4x4 * 4x4 (rank 49 = 7*7).
    Uses 7 outer Strassen products, each with inner Strassen 2x2*2x2 (rank 7).
    """
    U2, V2, W2, inner_rank = _strassen_2x2_2x2()
    # inner_rank = 7
    
    outer_rank = 7
    total_rank = outer_rank * inner_rank  # 49
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(16)]
    W = [[0]*total_rank for _ in range(16)]
    
    # Define outer Strassen products
    # Each product: (X_coeffs, Y_coeffs, W_coeffs)
    # X_coeffs: dict mapping A block entry (bi, bj) to coefficient
    #   where (bi, bj) means block row bi, block col bj, and within block (ri, rj)
    #   Global U row = bi*8 + bj*4 + ri*2 + rj  ... wait, let me re-index.
    
    # A is 4x4, indexed as A[i][j] where i,j in 0..3
    # Block A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # U row index = i*4 + j
    
    # For outer product Mk, Xk is a 2x2 matrix where:
    # Xk[0][0], Xk[0][1], Xk[1][0], Xk[1][1] are linear combos of A entries
    # Similarly for Yk with B entries
    # Pk = Xk * Yk is 2x2, and Pk[r][c] contributes to C block entries
    
    # Let me define each outer product as:
    # X_entries: list of 4 (coefficient, U_row) pairs for Xk[0][0], Xk[0][1], Xk[1][0], Xk[1][1]
    #   Actually, Xk[ri][rj] = sum of coeff * A[i][j] for various (i,j)
    #   I'll represent as: X_map[ri][rj] = list of (coeff, u_row)
    
    # Y_entries: similarly for Yk with V rows
    # W_map[ri][rj] = list of (coeff, w_row) for the C block contribution
    
    # Outer products:
    # M1 = (A11 - A22)(B21 + B22)
    #   X1[0][0] = A[0][0] - A[2][2]  -> U[0]=1, U[12]=-1
    #   X1[0][1] = A[0][1] - A[2][3]  -> U[1]=1, U[13]=-1
    #   X1[1][0] = A[1][0] - A[3][0]  -> U[4]=1, U[8]=-1
    #   X1[1][1] = A[1][1] - A[3][1]  -> U[5]=1, U[9]=-1
    #   Y1[0][0] = B[2][0] + B[3][0]  -> V[8]=1, V[12]=1
    #   Y1[0][1] = B[2][1] + B[3][1]  -> V[9]=1, V[13]=1
    #   Y1[1][0] = B[2][2] + B[3][2]  -> V[10]=1, V[14]=1
    #   Y1[1][1] = B[2][3] + B[3][3]  -> V[11]=1, V[15]=1
    #   P1 contributes: C11 (+1), C21 (+1), C22 (-1)
    #   C11 = C[0:2, 0:2], C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]
    #   P1[0][0] -> C[0][0] (W[0]), C[2][0] (W[8]), C[2][2] (W[12]) with coeffs +1,+1,-1
    #   P1[0][1] -> C[0][1] (W[1]), C[2][1] (W[9]), C[2][3] (W[13]) with coeffs +1,+1,-1
    #   P1[1][0] -> C[1][0] (W[4]), C[3][0] (W[10]), C[3][2] (W[14]) with coeffs +1,+1,-1
    #   P1[1][1] -> C[1][1] (W[5]), C[3][1] (W[11]), C[3][3] (W[15]) with coeffs +1,+1,-1
    
    # This is very tedious. Let me define a helper structure.
    
    # I'll define each outer product as a tuple of:
    # (X_map, Y_map, W_map)
    # X_map: 2x2 list of lists of (coeff, u_row)
    # Y_map: 2x2 list of lists of (coeff, v_row)
    # W_map: 2x2 list of lists of (coeff, w_row)
    
    def make_X(A_coeffs):
        """A_coeffs: 2x2 list of lists of (coeff, i, j) for A[i][j]"""
        result = [[[] for _ in range(2)] for _ in range(2)]
        for ri in range(2):
            for rj in range(2):
                for (coeff, i, j) in A_coeffs[ri][rj]:
                    result[ri][rj].append((coeff, i*4+j))
        return result
    
    def make_Y(B_coeffs):
        result = [[[] for _ in range(2)] for _ in range(2)]
        for ri in range(2):
            for rj in range(2):
                for (coeff, i, j) in B_coeffs[ri][rj]:
                    result[ri][rj].append((coeff, i*4+j))
        return result
    
    def make_W(C_coeffs):
        """C_coeffs: list of (coeff, block_row, block_col) where block is 2x2"""
        result = [[[] for _ in range(2)] for _ in range(2)]
        for (coeff, bi, bj) in C_coeffs:
            for ri in range(2):
                for rj in range(2):
                    i = bi*2 + ri
                    j = bj*2 + rj
                    result[ri][rj].append((coeff, i*4+j))
        return result
    
    # M1 = (A11 - A22)(B21 + B22)
    X1 = make_X([
        [[(1,0,0),(-1,2,2)],[(1,0,1),(-1,2,3)]],
        [[(1,1,0),(-1,3,0)],[(1,1,1),(-1,3,1)]]
    ])
    Y1 = make_Y([
        [[(1,2,0),(1,3,0)],[(1,2,1),(1,3,1)]],
        [[(1,2,2),(1,3,2)],[(1,2,3),(1,3,3)]]
    ])
    W1 = make_W([(1,0,0),(1,1,0),(-1,1,1)])  # C11, C21, C22
    
    # M2 = (A11 + A12) B21
    X2 = make_X([
        [[(1,0,0),(1,0,2)],[(1,0,1),(1,0,3)]],
        [[(1,1,0),(1,1,2)],[(1,1,1),(1,1,3)]]
    ])
    Y2 = make_Y([
        [[(1,2,0)],[(1,2,1)]],
        [[(1,3,0)],[(1,3,1)]]
    ])
    W2 = make_W([(1,0,1)])  # C12
    
    # M3 = A22 (B12 - B22)
    X3 = make_X([
        [[(1,2,2)],[(1,2,3)]],
        [[(1,3,2)],[(1,3,3)]]
    ])
    Y3 = make_Y([
        [[(1,0,2),(-1,2,2)],[(1,0,3),(-1,2,3)]],
        [[(1,1,2),(-1,3,2)],[(1,1,3),(-1,3,3)]]
    ])
    W3 = make_W([(1,1,0),(1,1,1)])  # C21, C22
    
    # M4 = (A21 + A22) B11
    X4 = make_X([
        [[(1,2,0),(1,2,2)],[(1,2,1),(1,2,3)]],
        [[(1,3,0),(1,3,2)],[(1,3,1),(1,3,3)]]
    ])
    Y4 = make_Y([
        [[(1,0,0)],[(1,0,1)]],
        [[(1,1,0)],[(1,1,1)]]
    ])
    W4 = make_W([(1,0,0),(-1,1,0),(1,1,1)])  # C11, C21, C22
    
    # M5 = (A11 + A22)(B11 + B22)
    X5 = make_X([
        [[(1,0,0),(1,2,2)],[(1,0,1),(1,2,3)]],
        [[(1,1,0),(1,3,0)],[(1,1,1),(1,3,1)]]
    ])
    Y5 = make_Y([
        [[(1,0,0),(1,2,2)],[(1,0,1),(1,2,3)]],
        [[(1,1,0),(1,3,0)],[(1,1,1),(1,3,1)]]
    ])
    W5 = make_W([(-1,0,0),(1,0,1),(1,1,0)])  # C11, C12, C21
    
    # M6 = (A12 - A22)(B11 + B12)
    X6 = make_X([
        [[(1,0,2),(-1,2,2)],[(1,0,3),(-1,2,3)]],
        [[(1,1,2),(-1,3,2)],[(1,1,3),(-1,3,3)]]
    ])
    Y6 = make_Y([
        [[(1,0,0),(1,0,2)],[(1,0,1),(1,0,3)]],
        [[(1,1,0),(1,1,2)],[(1,1,1),(1,1,3)]]
    ])
    W6 = make_W([(-1,0,1),(1,1,1)])  # C12, C22
    
    # M7 = (A21 - A11)(B12 + B22)
    X7 = make_X([
        [[(1,2,0),(-1,0,0)],[(1,2,1),(-1,0,1)]],
        [[(1,3,0),(-1,1,0)],[(1,3,1),(-1,1,1)]]
    ])
    Y7 = make_Y([
        [[(1,0,2),(1,2,2)],[(1,0,3),(1,2,3)]],
        [[(1,1,2),(1,3,2)],[(1,1,3),(1,3,3)]]
    ])
    W7 = make_W([(1,0,0),(-1,0,1)])  # C11, C12
    
    outer_products = [(X1,Y1,W1),(X2,Y2,W2),(X3,Y3,W3),(X4,Y4,W4),(X5,Y5,W5),(X6,Y6,W6),(X7,Y7,W7)]
    
    # Now combine: for each outer product k and inner term t, create a full term
    for k, (X_map, Y_map, W_map) in enumerate(outer_products):
        for t in range(inner_rank):
            col = k * inner_rank + t
            # Inner Strassen term t:
            # U: inner_U[ri*2+rj][t] for X[ri][rj]
            # V: inner_V[ri*2+rj][t] for Y[ri][rj]
            # W: inner_W[ri*2+rj][t] for P[ri][rj]
            
            for ri in range(2):
                for rj in range(2):
                    # X contribution
                    inner_u = U2[ri*2+rj][t]
                    if inner_u != 0:
                        for (coeff, u_row) in X_map[ri][rj]:
                            U[u_row][col] += coeff * inner_u
                    
                    # Y contribution
                    inner_v = V2[ri*2+rj][t]
                    if inner_v != 0:
                        for (coeff, v_row) in Y_map[ri][rj]:
                            V[v_row][col] += coeff * inner_v
                    
                    # W contribution
                    inner_w = W2[ri*2+rj][t]
                    if inner_w != 0:
                        for (coeff, w_row) in W_map[ri][rj]:
                            W[w_row][col] += coeff * inner_w
    
    return U, V, W, total_rank
