"""Rewritten program: Strassen-based rank-28 decomposition for 2x4 * 4x4 matrix multiplication."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4

    # The 7 rank-1 terms for 2x2 * 2x2 Strassen decomposition.
    # Each term is (u_vec, v_vec, w_vec) where:
    #   u_vec has 4 entries indexed by (r, c) in {0,1}^2 -> u[r*2+c]
    #   v_vec has 4 entries indexed by (r, c) in {0,1}^2 -> v[r*2+c]
    #   w_vec has 4 entries indexed by (k, i) in {0,1}^2 -> w[k*2+i]
    #
    # The Strassen 7 multiplications for 2x2*2x2:
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    #
    # Output:
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    #
    # Each Mi is a product of a linear form in X and a linear form in Y.
    # In the tensor decomposition, each Mi corresponds to ONE rank-1 term where:
    #   u_c is the coefficient vector for X (the linear form)
    #   v_c is the coefficient vector for Y (the linear form)
    #   w_c distributes Mi to the output positions with appropriate signs.
    
    # Term 1: M1 = (x00+x11)(y00+y11)
    # u1 = [1, 0, 0, 1]  (x00 + x11)
    # v1 = [1, 0, 0, 1]  (y00 + y11)
    # M1 contributes: Z00 += M1, Z11 += M1
    # w1 indexed by [k*2+i]: w[0*2+0]=1 (Z00), w[0*2+1]=0, w[1*2+0]=0, w[1*2+1]=1 (Z11)
    # w1 = [1, 0, 0, 1]
    t1 = ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1])
    
    # Term 2: M2 = (x10+x11)y00
    # u2 = [0, 0, 1, 1]  (x10 + x11)
    # v2 = [1, 0, 0, 0]  (y00)
    # M2 contributes: Z10 += M2, Z11 -= M2
    # w2: w[0*2+0]=0, w[0*2+1]=0, w[1*2+0]=1, w[1*2+1]=-1
    # w2 = [0, 0, 1, -1]
    t2 = ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1])
    
    # Term 3: M3 = x00(y01-y11)
    # u3 = [1, 0, 0, 0]  (x00)
    # v3 = [0, 1, 0, -1]  (y01 - y11)
    # M3 contributes: Z01 += M3, Z11 += M3
    # w3: w[0*2+0]=0, w[0*2+1]=1, w[1*2+0]=0, w[1*2+1]=1
    # w3 = [0, 1, 0, 1]
    t3 = ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1])
    
    # Term 4: M4 = x11(y10-y00)
    # u4 = [0, 0, 0, 1]  (x11)
    # v4 = [0, 0, 1, -1]  (y10 - y00)
    # M4 contributes: Z00 += M4, Z10 += M4
    # w4: w[0*2+0]=1, w[0*2+1]=0, w[1*2+0]=1, w[1*2+1]=0
    # w4 = [1, 0, 1, 0]
    t4 = ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0])
    
    # Term 5: M5 = (x00+x01)y11
    # u5 = [1, 1, 0, 0]  (x00 + x01)
    # v5 = [0, 0, 0, 1]  (y11)
    # M5 contributes: Z00 -= M5, Z01 += M5
    # w5: w[0*2+0]=-1, w[0*2+1]=1, w[1*2+0]=0, w[1*2+1]=0
    # w5 = [-1, 1, 0, 0]
    t5 = ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0])
    
    # Term 6: M6 = (x10-x00)(y00+y01)
    # u6 = [0, 0, 1, -1]  (x10 - x00)
    # v6 = [1, 1, 0, 0]  (y00 + y01)
    # M6 contributes: Z11 += M6
    # w6: w[0*2+0]=0, w[0*2+1]=0, w[1*2+0]=0, w[1*2+1]=1
    # w6 = [0, 0, 0, 1]
    t6 = ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1])
    
    # Term 7: M7 = (x01-x11)(y10+y11)
    # u7 = [0, 1, 0, -1]  (x01 - x11)
    # v7 = [0, 0, 1, 1]  (y10 + y11)
    # M7 contributes: Z00 += M7
    # w7: w[0*2+0]=1, w[0*2+1]=0, w[1*2+0]=0, w[1*2+1]=0
    # w7 = [1, 0, 0, 0]
    t7 = ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    
    strassen_terms = [t1, t2, t3, t4, t5, t6, t7]
    
    # Now apply to the 4 blocks of the 2x4 * 4x4 problem.
    # A is 2x4, B is 4x4, C is 2x4.
    # C[i][k] = sum_{j=0}^{3} A[i][j] * B[j][k]
    #
    # Split into 4 blocks of 2x2 * 2x2:
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> contributes to C[0:2, 0:2]
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> contributes to C[0:2, 0:2]
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> contributes to C[0:2, 2:4]
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> contributes to C[0:2, 2:4]
    
    # For each block, the local 2x2 indices map to global indices:
    # A block at (a_row_off, a_col_off): A[a_row_off+r][a_col_off+c]
    # B block at (b_row_off, b_col_off): B[b_row_off+r][b_col_off+c]
    # C contribution at (c_row_off, c_col_off): C[c_row_off+r][c_col_off+c]
    #
    # Global U index for A[i][j]: i*m + j = i*4 + j
    # Global V index for B[j][k]: j*p + k = j*4 + k
    # Global W index for C[i][k]: k*n + i = k*2 + i
    
    blocks = [
        (0, 0, 0, 0),  # a_row_off, a_col_off, b_row_off, b_col_off
        (0, 2, 2, 0),
        (0, 0, 0, 2),
        (0, 2, 2, 2),
    ]
    
    rank = 4 * 7  # 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for (a_ro, a_co, b_ro, b_co) in blocks:
        for (u_loc, v_loc, w_loc) in strassen_terms:
            # u_loc[r*2+c] corresponds to A[a_ro+r][a_co+c]
            for r in range(2):
                for c in range(2):
                    u_val = u_loc[r * 2 + c]
                    if u_val != 0:
                        u_global = (a_ro + r) * m + (a_co + c)
                        U[u_global][col] = u_val
            
            # v_loc[r*2+c] corresponds to B[b_ro+r][b_co+c]
            for r in range(2):
                for c in range(2):
                    v_val = v_loc[r * 2 + c]
                    if v_val != 0:
                        v_global = (b_ro + r) * p + (b_co + c)
                        V[v_global][col] = v_val
            
            # w_loc[k*2+i] corresponds to C[i][k] where i is row, k is col
            # In our block, C row offset is a_ro, C col offset is b_co
            for k in range(2):
                for i in range(2):
                    w_val = w_loc[k * 2 + i]
                    if w_val != 0:
                        w_global = (b_co + k) * n + (a_ro + i)
                        W[w_global][col] = w_val
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
