"""Strassen-based rank-28 tensor decomposition for 4x4 matrix multiplication."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n = m = p = 4
    
    # Use Strassen at 4x4 level with schoolbook at 2x2 level.
    # Rank = 7 (Strassen) * 4 (schoolbook 2x2) = 28
    
    rank = 28
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Block definitions (0-indexed block rows/cols):
    # Block (br, bc) covers A rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # Global A index for block (br,bc) local (lr,lc): (2*br+lr)*4 + (2*bc+lc)
    # Same for B.
    # C block (cr,cc) local (lr,lc): global C index (2*cr+lr, 2*cc+lc)
    # W index for C[i,k]: k*4 + i
    
    # Strassen terms: each has X (linear combo of A blocks), Y (linear combo of B blocks),
    # and output mapping (C block -> sign).
    # X = sum_{br,bc} xcoef[br,bc] * A_block(br,bc)
    # Y = sum_{br,bc} ycoef[br,bc] * B_block(br,bc)
    
    strassen = [
        # S1: X=A11+A22, Y=B11+B22, out: C11+1, C22+1
        {'X': {((0,0), 1), ((1,1), 1)}, 'Y': {((0,0), 1), ((1,1), 1)}, 
         'out': {((0,0), 1), ((1,1), 1)}},
        # S2: X=A21+A22, Y=B11, out: C12+1
        {'X': {((1,0), 1), ((1,1), 1)}, 'Y': {((0,0), 1)}, 
         'out': {((0,1), 1)}},
        # S3: X=A11, Y=B12-B22, out: C12+1, C22-1
        {'X': {((0,0), 1)}, 'Y': {((0,1), 1), ((1,1), -1)}, 
         'out': {((0,1), 1), ((1,1), -1)}},
        # S4: X=A22, Y=B21-B11, out: C11+1, C22+1
        {'X': {((1,1), 1)}, 'Y': {((1,0), 1), ((0,0), -1)}, 
         'out': {((0,0), 1), ((1,1), 1)}},
        # S5: X=A11+A12, Y=B22, out: C11-1, C21+1
        {'X': {((0,0), 1), ((0,1), 1)}, 'Y': {((1,1), 1)}, 
         'out': {((0,0), -1), ((1,0), 1)}},
        # S6: X=A21-A11, Y=B11+B12, out: C21+1
        {'X': {((1,0), 1), ((0,0), -1)}, 'Y': {((0,0), 1), ((0,1), 1)}, 
         'out': {((1,0), 1)}},
        # S7: X=A12-A22, Y=B21+B22, out: C11-1, C22+1
        {'X': {((0,1), 1), ((1,1), -1)}, 'Y': {((1,0), 1), ((1,1), 1)}, 
         'out': {((0,0), -1), ((1,1), 1)}},
    ]
    
    col = 0
    for s in strassen:
        X_coefs = s['X']  # {(br,bc): coef}
        Y_coefs = s['Y']
        out_coefs = s['out']  # {(cr,cc): sign}
        
        # X is 2x2: X[lr][lc] = sum_{(br,bc)} xcoef * A_block(br,bc)[lr][lc]
        # Y is 2x2: Y[jr][jc] = sum_{(br,bc)} ycoef * B_block(br,bc)[jr][jc]
        # Result R[lr][jc] = sum_j X[lr][j] * Y[j][jc]
        
        for lr in range(2):
            for jc in range(2):
                # R[lr][jc] = sum_j X[lr][j] * Y[j][jc]
                # = sum_j sum_{bx} xcoef[bx] * A_block(bx)[lr][j] * sum_{by} ycoef[by] * B_block(by)[j][jc]
                # = sum_{bx,by,j} xcoef[bx]*ycoef[by] * A_block(bx)[lr][j] * B_block(by)[j][jc]
                
                for j in range(2):
                    for (bx, xc) in X_coefs.items():
                        for (by, yc) in Y_coefs.items():
                            coeff = xc * yc
                            
                            # A element: block bx=(brx,bcx), local [lr][j]
                            a_row = 2*bx[0] + lr
                            a_col = 2*bx[1] + j
                            a_idx = a_row * 4 + a_col
                            
                            # B element: block by=(bry,bcy), local [j][jc]
                            b_row = 2*by[0] + j
                            b_col = 2*by[1] + jc
                            b_idx = b_row * 4 + b_col
                            
                            U[a_idx][col] += coeff
                            V[b_idx][col] += coeff
                            
                            # Output: R[lr][jc] goes to each C block in out_coefs
                            for (cblock, sign) in out_coefs.items():
                                c_row = 2*cblock[0] + lr
                                c_col = 2*cblock[1] + jc
                                w_idx = c_col * 4 + c_row  # W[k*n+i] where k=c_col, i=c_row
                                W[w_idx][col] += coeff * sign
                            
                            col += 1
    # col should be 28
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
