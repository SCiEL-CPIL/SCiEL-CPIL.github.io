import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)] # 10 x 54
    V = [[0] * rank for _ in range(m * p)] # 30 x 54
    W = [[0] * rank for _ in range(p * n)] # 12 x 54
    
    # Strassen factors for 2x2x2
    # Each factor is a tuple (u, v, w) where u, v, w are flattened 2x2 matrices [a,b,c,d]
    # u corresponds to A, v to B, w to C.
    # Indices in 2x2: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    
    # M1: (A11+A22)(B11+B22)
    # u1: [1, 0, 0, 1]
    # v1: [1, 0, 0, 1]
    # w1: [1, 0, 0, 1] (C11+=1, C22+=1)
    
    # M2: (A21+A22)B11
    # u2: [0, 0, 1, 1]
    # v2: [1, 0, 0, 0]
    # w2: [0, 0, 1, 0] (C21+=1)
    
    # M3: A11(B21-B22)
    # u3: [1, 0, 0, 0]
    # v3: [0, 0, 1, -1]
    # w3: [0, 1, 0, 1] (C12+=1, C22+=1)
    
    # M4: A22(B12-B11)
    # u4: [0, 0, 0, 1]
    # v4: [-1, 1, 0, 0]
    # w4: [1, 0, 1, 0] (C11+=1, C21+=1)
    
    # M5: (A11+A12)B22
    # u5: [1, 1, 0, 0]
    # v5: [0, 0, 0, 1]
    # w5: [-1, 1, 0, 0] (C11-=1, C12+=1)
    
    # M6: (A21-A11)(B11+B12)
    # u6: [-1, 0, 1, 0]
    # v6: [1, 1, 0, 0]
    # w6: [0, 0, 0, 1] (C22+=1)
    
    # M7: (A12-A22)(B21+B22)
    # u7: [0, 1, -1, 0]
    # v7: [0, 0, 1, 1]
    # w7: [1, 0, 0, 1] (C11+=1, C22+=1)
    
    strassen_factors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 1])
    ]
    
    def add_strassen_block(u_row_off, u_col_off, v_row_off, v_col_off, w_row_off, w_col_off, current_col):
        """
        Adds a 2x2x2 Strassen decomposition to the global matrices.
        u_row_off: offset for rows of A (2x2 block)
        u_col_off: offset for cols of A (2x2 block)
        v_row_off: offset for rows of B (2x2 block)
        v_col_off: offset for cols of B (2x2 block)
        w_row_off: offset for rows of C (2x2 block) - Note: C is 2x2 here
        w_col_off: offset for cols of C (2x2 block)
        """
        for r in range(7):
            u, v, w = strassen_factors[r]
            for i in range(2):
                for j in range(2):
                    # U index: (u_row_off + i) * m + (u_col_off + j)
                    U[(u_row_off + i) * m + (u_col_off + j)][current_col] = u[i*2+j]
                    
                    # V index: (v_row_off + i) * p + (v_col_off + j)
                    V[(v_row_off + i) * p + (v_col_off + j)][current_col] = v[i*2+j]
                    
                    # W index: (w_col_off + j) * n + (w_row_off + i)
                    # Note: W[k*n+i, r] where k is col of C, i is row of C.
                    # In the 2x2 block, local col is j, local row is i.
                    # Global col of C = w_col_off + j.
                    # Global row of C = w_row_off + i.
                    W[(w_col_off + j) * n + (w_row_off + i)][current_col] = w[i*2+j]
            current_col += 1
        return current_col

    col = 0
    
    # Term 1: 2x4x4
    # A cols 0-3, B rows 0-3 cols 0-3, C rows 0-1 cols 0-3
    # Split A into A_L (cols 0,1) and A_R (cols 2,3)
    # Split B into B_LL (rows 0,1, cols 0,1), B_LR (rows 0,1, cols 2,3), B_RL (rows 2,3, cols 0,1), B_RR (rows 2,3, cols 2,3)
    # C_L (cols 0,1) = A_L B_LL + A_R B_RL
    # C_R (cols 2,3) = A_L B_LR + A_R B_RR
    
    # 1. A_L * B_LL -> C_L (part 1)
    # u: rows 0,1 cols 0,1. v: rows 0,1 cols 0,1. w: rows 0,1 cols 0,1.
    col = add_strassen_block(0, 0, 0, 0, 0, 0, col)
    
    # 2. A_R * B_RL -> C_L (part 2)
    # u: rows 0,1 cols 2,3. v: rows 2,3 cols 0,1. w: rows 0,1 cols 0,1.
    col = add_strassen_block(0, 2, 2, 0, 0, 0, col)
    
    # 3. A_L * B_LR -> C_R (part 1)
    # u: rows 0,1 cols 0,1. v: rows 0,1 cols 2,3. w: rows 0,1 cols 2,3.
    col = add_strassen_block(0, 0, 0, 2, 0, 2, col)
    
    # 4. A_R * B_RR -> C_R (part 2)
    # u: rows 0,1 cols 2,3. v: rows 2,3 cols 2,3. w: rows 0,1 cols 2,3.
    col = add_strassen_block(0, 2, 2, 2, 0, 2, col)
    
    # Term 2: 2x4x2
    # A cols 0-3, B rows 0-3 cols 4-5, C rows 0-1 cols 4-5
    # C = A_L B_L + A_R B_R
    # B_L: rows 0,1 cols 4,5. B_R: rows 2,3 cols 4,5.
    
    # 5. A_L * B_L -> C (part 1)
    # u: rows 0,1 cols 0,1. v: rows 0,1 cols 4,5. w: rows 0,1 cols 4,5.
    col = add_strassen_block(0, 0, 0, 4, 0, 4, col)
    
    # 6. A_R * B_R -> C (part 2)
    # u: rows 0,1 cols 2,3. v: rows 2,3 cols 4,5. w: rows 0,1 cols 4,5.
    col = add_strassen_block(0, 2, 2, 4, 0, 4, col)
    
    # Term 3: 2x1x6
    # A col 4, B row 4 cols 0-5, C rows 0-1 cols 0-5
    # C[i, k] = A[i, 4] * B[4, k]
    for i in range(2):
        for k in range(6):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
