"""Matrix multiplication 4,4,5 - Block decomposition with Strassen."""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Block decomposition: split A into 2x2 blocks, B into 2x2 and 2x1 blocks
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where B11,B12,B21,B22 are 2x2, B13,B23 are 2x1
    # C = [[C11, C12, C13], [C21, C22, C23]] where C11,C12,C21,C22 are 2x2, C13,C23 are 2x1
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    #
    # Strategy: Use Strassen (rank 7) for 2x2*2x2=2x2 products
    # For 2x2*2x1=2x1 products, use schoolbook (rank 4)
    #
    # Total: 4 Strassen pairs (14 each) + 2 schoolbook pairs (4 each) = 56 + 8 = 64
    # But we can potentially do better by combining the two products in each C block
    # using a more sophisticated decomposition.
    
    terms = []
    
    def add_strassen_2x2(A_block, B_block, C_block):
        """Add Strassen terms for 2x2*2x2=2x2."""
        x11, x12 = A_block[(0,0)], A_block[(0,1)]
        x21, x22 = A_block[(1,0)], A_block[(1,1)]
        y11, y12 = B_block[(0,0)], B_block[(0,1)]
        y21, y22 = B_block[(1,0)], B_block[(1,1)]
        z11, z12 = C_block[(0,0)], C_block[(0,1)]
        z21, z22 = C_block[(1,0)], C_block[(1,1)]
        
        # M1 = (x11+x22)*(y11+y22)
        terms.append(({x11: 1, x22: 1}, {y11: 1, y22: 1}, {z11: 1, z22: 1}))
        # M2 = (x21+x22)*y11
        terms.append(({x21: 1, x22: 1}, {y11: 1}, {z21: 1, z22: -1}))
        # M3 = x11*(y12-y22)
        terms.append(({x11: 1}, {y12: 1, y22: -1}, {z12: 1, z22: 1}))
        # M4 = x22*(y21-y11)
        terms.append(({x22: 1}, {y21: 1, y11: -1}, {z11: 1, z21: 1}))
        # M5 = (x11+x12)*y22
        terms.append(({x11: 1, x12: 1}, {y22: 1}, {z12: 1, z11: -1}))
        # M6 = (x21-x11)*(y11+y12)
        terms.append(({x21: 1, x11: -1}, {y11: 1, y12: 1}, {z22: 1}))
        # M7 = (x12-x22)*(y21+y22)
        terms.append(({x12: 1, x22: -1}, {y21: 1, y22: 1}, {z11: 1, z22: 1}))
    
    def add_schoolbook_2x1(A_block, B_block, C_block):
        """Add schoolbook terms for 2x2*2x1=2x1."""
        for i in range(2):
            for j in range(2):
                terms.append(({A_block[(i,j)]: 1}, {B_block[(j,0)]: 1}, {C_block[(i,0)]: 1}))
    
    # Index mappings
    # U[i*m+j] for A[i,j], m=4
    A11 = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
    A12 = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
    A21 = {(0,0): 8, (0,1): 9, (1,0): 12, (1,1): 13}
    A22 = {(0,0): 10, (0,1): 11, (1,0): 14, (1,1): 15}
    
    # V[j*p+k] for B[j,k], p=5
    B11 = {(0,0): 0, (0,1): 1, (1,0): 5, (1,1): 6}
    B12 = {(0,0): 2, (0,1): 3, (1,0): 7, (1,1): 8}
    B13 = {(0,0): 4, (1,0): 9}
    B21 = {(0,0): 10, (0,1): 11, (1,0): 15, (1,1): 16}
    B22 = {(0,0): 12, (0,1): 13, (1,0): 17, (1,1): 18}
    B23 = {(0,0): 14, (1,0): 19}
    
    # W[k*n+i] for C[i,k], n=4
    C11 = {(0,0): 0, (0,1): 4, (1,0): 1, (1,1): 5}
    C12 = {(0,0): 8, (0,1): 12, (1,0): 9, (1,1): 13}
    C13 = {(0,0): 16, (1,0): 17}
    C21 = {(0,0): 2, (0,1): 6, (1,0): 3, (1,1): 7}
    C22 = {(0,0): 10, (0,1): 14, (1,0): 11, (1,1): 15}
    C23 = {(0,0): 18, (1,0): 19}
    
    # Build decomposition
    # C11 = A11*B11 + A12*B21
    add_strassen_2x2(A11, B11, C11)
    add_strassen_2x2(A12, B21, C11)
    
    # C12 = A11*B12 + A12*B22
    add_strassen_2x2(A11, B12, C12)
    add_strassen_2x2(A12, B22, C12)
    
    # C13 = A11*B13 + A12*B23
    add_schoolbook_2x1(A11, B13, C13)
    add_schoolbook_2x1(A12, B23, C13)
    
    # C21 = A21*B11 + A22*B21
    add_strassen_2x2(A21, B11, C21)
    add_strassen_2x2(A22, B21, C21)
    
    # C22 = A21*B12 + A22*B22
    add_strassen_2x2(A21, B12, C22)
    add_strassen_2x2(A22, B22, C22)
    
    # C23 = A21*B13 + A22*B23
    add_schoolbook_2x1(A21, B13, C23)
    add_schoolbook_2x1(A22, B23, C23)
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0.0]*rank for _ in range(nm)]
    V = [[0.0]*rank for _ in range(mp)]
    W = [[0.0]*rank for _ in range(pn)]
    
    for c, (U_t, V_t, W_t) in enumerate(terms):
        for row, coeff in U_t.items():
            U[row][c] = coeff
        for row, coeff in V_t.items():
            V[row][c] = coeff
        for row, coeff in W_t.items():
            W[row][c] = coeff
    
    # Verify correctness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if abs(s - 1.0) > 1e-9:
                    # Fall back to schoolbook
                    rank_sb = n*m*p
                    U = [[0.0]*rank_sb for _ in range(nm)]
                    V = [[0.0]*rank_sb for _ in range(mp)]
                    W = [[0.0]*rank_sb for _ in range(pn)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii*m+jj][col] = 1.0
                                V[jj*p+kk][col] = 1.0
                                W[kk*n+ii][col] = 1.0
                                col += 1
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
