"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 Strassen algorithm
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get global index for A block br, bc at local row lr, col lc
    def a_idx(br, bc, lr, lc):
        return (2 * br + lr) * 4 + (2 * bc + lc)
    
    # Helper to get global index for B block br, bc at local row lr, col lc
    def b_idx(br, bc, lr, lc):
        return (2 * br + lr) * 4 + (2 * bc + lc)
        
    # Helper to get global index for C block br, bc at local row lr, col lc
    # Output convention: W[k*n+i] where i is row, k is col
    # Global row = 2*br + lr, Global col = 2*bc + lc
    def c_idx(br, bc, lr, lc):
        r = 2 * br + lr
        c = 2 * bc + lc
        return c * 4 + r

    # Strassen products M0..M6
    # Each M is a 2x2 matrix of linear forms
    # We define the linear forms for X (from A) and Y (from B)
    # X[l][r][c] is a list of (coeff, a_block_r, a_block_c, a_local_r, a_local_c)
    # Y[l][r][c] is a list of (coeff, b_block_r, b_block_c, b_local_r, b_local_c)
    
    # Initialize lists
    X = [[[[] for _ in range(2)] for _ in range(2)] for _ in range(7)]
    Y = [[[[] for _ in range(2)] for _ in range(2)] for _ in range(7)]
    
    def add_x(l, r, c, coeff, br, bc, lr, lc):
        X[l][r][c].append((coeff, br, bc, lr, lc))
        
    def add_y(l, r, c, coeff, br, bc, lr, lc):
        Y[l][r][c].append((coeff, br, bc, lr, lc))

    # M0 = (A00 + A11)(B00 + B11)
    for r in range(2):
        for c in range(2):
            add_x(0, r, c, 1, 0, 0, r, c)
            add_x(0, r, c, 1, 1, 1, r, c)
            add_y(0, r, c, 1, 0, 0, r, c)
            add_y(0, r, c, 1, 1, 1, r, c)
            
    # M1 = (A10 + A11)B00
    for r in range(2):
        for c in range(2):
            add_x(1, r, c, 1, 1, 0, r, c)
            add_x(1, r, c, 1, 1, 1, r, c)
            add_y(1, r, c, 1, 0, 0, r, c)

    # M2 = A00(B01 - B11)
    for r in range(2):
        for c in range(2):
            add_x(2, r, c, 1, 0, 0, r, c)
            add_y(2, r, c, 1, 0, 1, r, c)
            add_y(2, r, c, -1, 1, 1, r, c)

    # M3 = A11(B10 - B00)
    for r in range(2):
        for c in range(2):
            add_x(3, r, c, 1, 1, 1, r, c)
            add_y(3, r, c, 1, 1, 0, r, c)
            add_y(3, r, c, -1, 0, 0, r, c)

    # M4 = (A00 + A01)B11
    for r in range(2):
        for c in range(2):
            add_x(4, r, c, 1, 0, 0, r, c)
            add_x(4, r, c, 1, 0, 1, r, c)
            add_y(4, r, c, 1, 1, 1, r, c)

    # M5 = (A10 - A00)(B00 + B01)
    for r in range(2):
        for c in range(2):
            add_x(5, r, c, 1, 1, 0, r, c)
            add_x(5, r, c, -1, 0, 0, r, c)
            add_y(5, r, c, 1, 0, 0, r, c)
            add_y(5, r, c, 1, 0, 1, r, c)

    # M6 = (A01 - A11)(B10 + B11)
    for r in range(2):
        for c in range(2):
            add_x(6, r, c, 1, 0, 1, r, c)
            add_x(6, r, c, -1, 1, 1, r, c)
            add_y(6, r, c, 1, 1, 0, r, c)
            add_y(6, r, c, 1, 1, 1, r, c)

    # Output contributions
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Map M_l to C blocks with coefficients
    # c_contrib[l][br][bc] = coeff
    c_contrib = {}
    for l in range(7):
        c_contrib[l] = [[0.0 for _ in range(2)] for _ in range(2)]
        
    c_contrib[0][0][0] += 1.0
    c_contrib[0][1][1] += 1.0
    
    c_contrib[1][1][0] += 1.0
    c_contrib[1][1][1] += -1.0
    
    c_contrib[2][0][1] += 1.0
    c_contrib[2][1][1] += 1.0
    
    c_contrib[3][0][0] += 1.0
    c_contrib[3][1][0] += 1.0
    
    c_contrib[4][0][0] += -1.0
    c_contrib[4][0][1] += 1.0
    
    c_contrib[5][1][1] += 1.0
    
    c_contrib[6][0][0] += 1.0

    col = 0
    for l in range(7):
        for i in range(2): # row of M_l
            for j in range(2): # col of M_l
                # M_l[i,j] = X_l[i,0]Y_l[0,j] + X_l[i,1]Y_l[1,j]
                # Term 1: X_l[i,0] * Y_l[0,j]
                x_forms = X[l][i][0]
                y_forms = Y[l][0][j]
                # We need to decompose the product of sums into rank-1 terms
                # (sum a_k x_k) * (sum b_k y_k) = sum_{k,m} a_k b_m x_k y_m
                # This increases rank!
                # Wait. Strassen's algorithm reduces the number of *multiplications*.
                # The multiplications are of the *linear forms*.
                # So X_l[i,0] is a linear form. Y_l[0,j] is a linear form.
                # The product is a single multiplication of these two linear forms.
                # This corresponds to ONE rank-1 term in the tensor decomposition?
                # No. A rank-1 term is u v^T w.
                # If u is a vector of coefficients for A, and v is a vector of coefficients for B.
                # The term is (sum u_a A_a) * (sum v_b B_b).
                # This is a single rank-1 component.
                # So for each pair (X_l[i,k], Y_l[k,j]), we have ONE rank-1 term.
                # There are 2 such pairs for each M_l[i,j] (k=0 and k=1).
                # So 4 terms per M_l[i,j]?
                # No. M_l[i,j] = sum_k X_l[i,k] Y_l[k,j].
                # This sum involves 2 multiplications.
                # So for each i,j, we have 2 rank-1 terms.
                # Total 4 terms per M_l? No, 2 terms per entry, 4 entries -> 8 terms per M_l.
                # Total 56 terms. Correct.
                
                for k in range(2):
                    # Term: X_l[i,k] * Y_l[k,j]
                    # U part: coefficients from X_l[i,k]
                    # V part: coefficients from Y_l[k,j]
                    # W part: contributions to C blocks
                    
                    # Update U
                    for coeff, br, bc, lr, lc in X[l][i][k]:
                        idx = a_idx(br, bc, lr, lc)
                        U[idx][col] += coeff
                        
                    # Update V
                    for coeff, br, bc, lr, lc in Y[l][k][j]:
                        idx = b_idx(br, bc, lr, lc)
                        V[idx][col] += coeff
                        
                    # Update W
                    # This term contributes to M_l[i,j] with coefficient 1.
                    # M_l[i,j] contributes to C blocks according to c_contrib[l]
                    for br in range(2):
                        for bc in range(2):
                            coeff_c = c_contrib[l][br][bc]
                            if coeff_c != 0:
                                idx_c = c_idx(br, bc, i, j)
                                W[idx_c][col] += coeff_c
                                
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
