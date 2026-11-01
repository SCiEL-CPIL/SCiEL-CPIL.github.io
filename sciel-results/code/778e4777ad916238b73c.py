import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49
    rank = 49
    
    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # We will fill these using the Strassen decomposition.
    # Indices:
    # A[i, j] -> u = i*m + j
    # B[j, k] -> v = j*p + k
    # C[i, k] -> w = k*n + i (Note: W[k*n+i, r])
    
    # Strassen 2x2 blocks
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    
    # We define the 7 block multiplications M1..M7
    # Each M is a 2x2 matrix.
    # We decompose each M into 7 scalar mults.
    
    # Helper to add a scalar multiplication term
    # x_coeffs: dict (i, j) -> coeff
    # y_coeffs: dict (j, k) -> coeff
    # c_coeffs: dict (i, k) -> coeff
    col = 0
    for l in range(7):
        # Define X and Y for the l-th Strassen step
        # These are 2x2 matrices of linear combinations of A and B blocks
        # We represent them as 4x4 matrices of coefficients for A and B
        # Actually, X is a linear combo of A blocks, Y is a linear combo of B blocks.
        
        # Strassen formulas for 2x2 block matrices
        if l == 0:
            # M1 = (A11 + A22)(B11 + B22)
            # X = A11 + A22, Y = B11 + B22
            x_blocks = [(0,0,1), (1,1,1)] # (row, col, coeff)
            y_blocks = [(0,0,1), (1,1,1)]
        elif l == 1:
            # M2 = (A21 + A22)B11
            x_blocks = [(1,0,1), (1,1,1)]
            y_blocks = [(0,0,1)]
        elif l == 2:
            # M3 = A11(B12 - B22)
            x_blocks = [(0,0,1)]
            y_blocks = [(0,1,1), (1,1,-1)]
        elif l == 3:
            # M4 = A22(B21 - B11)
            x_blocks = [(1,1,1)]
            y_blocks = [(1,0,1), (0,0,-1)]
        elif l == 4:
            # M5 = (A11 + A12)B22
            x_blocks = [(0,0,1), (0,1,1)]
            y_blocks = [(1,1,1)]
        elif l == 5:
            # M6 = (A21 - A11)(B11 + B12)
            x_blocks = [(1,0,1), (0,0,-1)]
            y_blocks = [(0,0,1), (0,1,1)]
        elif l == 6:
            # M7 = (A12 - A22)(B21 + B22)
            x_blocks = [(0,1,1), (1,1,-1)]
            y_blocks = [(1,0,1), (1,1,1)]
        else:
            continue
            
        # Now decompose the product of these 2x2 block matrices into scalar multiplications
        # X is 2x2, Y is 2x2.
        # We apply Strassen to X*Y? No, X and Y are already sums.
        # We just need to multiply the 2x2 matrices X and Y.
        # X = [[x00, x01], [x10, x11]]
        # Y = [[y00, y01], [y10, y11]]
        # Product P = XY.
        # We can use the 7 Strassen multiplications for P.
        
        # Construct X and Y as 2x2 matrices of "block coefficients"
        # Each element is a linear combo of A[i,j] or B[j,k]
        
        # Let's represent X elements as lists of (i, j, coeff)
        X = [[0]*2 for _ in range(2)]
        Y = [[0]*2 for _ in range(2)]
        
        # Fill X
        for r in range(2):
            for c in range(2):
                X[r][c] = []
        for r in range(2):
            for c in range(2):
                Y[r][c] = []
                
        for (br, bc, coeff) in x_blocks:
            # br, bc are block indices (0,0) to (1,1)
            # Expand to scalar indices
            for dr in range(2):
                for dc in range(2):
                    i = br*2 + dr
                    j = bc*2 + dc
                    X[br][bc].append((i, j, coeff))
                    
        for (br, bc, coeff) in y_blocks:
            for dr in range(2):
                for dc in range(2):
                    j = br*2 + dr
                    k = bc*2 + dc
                    Y[br][bc].append((j, k, coeff))

        # Now compute the 7 scalar multiplications for the product of 2x2 matrices X and Y
        # Standard Strassen for 2x2:
        # S1 = X[0][0] + X[1][1]
        # S2 = X[1][0] + X[1][1]
        # S3 = X[0][0]
        # S4 = X[1][1]
        # S5 = X[0][0] + X[0][1]
        # S6 = X[1][0] - X[0][0]
        # S7 = X[0][1] - X[1][1]
        
        # T1 = Y[0][0] + Y[1][1]
        # T2 = Y[0][0]
        # T3 = Y[0][1] - Y[1][1]
        # T4 = Y[1][0] - Y[0][0]
        # T5 = Y[1][1]
        # T6 = Y[0][0] + Y[0][1]
        # T7 = Y[1][0] + Y[1][1]
        
        # M1 = S1 T1
        # M2 = S2 T2
        # M3 = S3 T3
        # M4 = S4 T4
        # M5 = S5 T5
        # M6 = S6 T6
        # M7 = S7 T7
        
        # P[0][0] = M1 - M4 + M5 - M7
        # P[0][1] = M3 + M7
        # P[1][0] = M2 + M4
        # P[1][1] = M1 - M2 + M3 + M6
        
        # We need to add 7 terms to U, V, W.
        # Each term is a product of a linear combo of A and a linear combo of B.
        
        def add_term(x_list, y_list, p_row, p_col, coeff):
            nonlocal col
            if col >= rank: return
            # x_list: list of (i, j, coeff_x)
            # y_list: list of (j, k, coeff_y)
            # The term contributes coeff * (sum x) * (sum y) to P[p_row][p_col]
            # P is a 2x2 block in C.
            # p_row, p_col are block indices (0,0) to (1,1).
            # Expand to scalar indices in C.
            for dr in range(2):
                for dc in range(2):
                    i = p_row*2 + dr
                    k = p_col*2 + dc
                    w_idx = k*n + i
                    W[w_idx, col] += coeff
                    for (i_x, j_x, c_x) in x_list:
                        u_idx = i_x*m + j_x
                        U[u_idx, col] += c_x
                    for (j_y, k_y, c_y) in y_list:
                        v_idx = j_y*p + k_y
                        V[v_idx, col] += c_y
            col += 1

        # Construct S and T lists
        # S1 = X[0][0] + X[1][1]
        S1 = X[0][0] + X[1][1]
        S2 = X[1][0] + X[1][1]
        S3 = X[0][0]
        S4 = X[1][1]
        S5 = X[0][0] + X[0][1]
        S6 = X[1][0] + [(-i, -j, -c) for (i, j, c) in X[0][0]] # X[1][0] - X[0][0]
        S7 = X[0][1] + [(-i, -j, -c) for (i, j, c) in X[1][1]] # X[0][1] - X[1][1]
        
        T1 = Y[0][0] + Y[1][1]
        T2 = Y[0][0]
        T3 = Y[0][1] + [(-i, -j, -c) for (i, j, c) in Y[1][1]] # Y[0][1] - Y[1][1]
        T4 = Y[1][0] + [(-i, -j, -c) for (i, j, c) in Y[0][0]] # Y[1][0] - Y[0][0]
        T5 = Y[1][1]
        T6 = Y[0][0] + Y[0][1]
        T7 = Y[1][0] + Y[1][1]
        
        # Add the 7 products
        # M1 = S1 T1 -> contributes to P[0][0] (-M4+M5-M7), P[1][1] (M1-M2+M3+M6)
        # Wait, the Strassen formulas for the product P are:
        # P[0][0] = M1 - M4 + M5 - M7
        # P[0][1] = M3 + M7
        # P[1][0] = M2 + M4
        # P[1][1] = M1 - M2 + M3 + M6
        
        # So M1 (S1*T1) contributes:
        # to P[0][0] with coeff 1
        # to P[1][1] with coeff 1
        add_term(S1, T1, 0, 0, 1)
        add_term(S1, T1, 1, 1, 1) # This is wrong. M1 is one term.
        
        # Correction:
        # Each of the 7 multiplications is a single rank-1 term in the 2x2 product.
        # But the 2x2 product is part of the 4x4 product.
        # The 7 multiplications are S1*T1, S2*T2, etc.
        # Each S*T is a scalar multiplication of two linear combinations.
        # So each S*T is ONE rank-1 term in the overall 4x4 decomposition?
        # No, S is a 2x2 matrix? No, S1 is a scalar (linear combo of A elements).
        # So S1*T1 is a scalar multiplication.
        # It contributes to multiple entries of P.
        # So for each of the 7 (S, T) pairs, we add ONE column to U, V, W.
        # And W will have non-zero entries in multiple places.
        
        # Let's redo the addition.
        # We have 7 terms for this block l.
        # Term 1: S1 * T1
        # Term 2: S2 * T2
        # ...
        
        terms = [
            (S1, T1, 1), # M1
            (S2, T2, 1), # M2
            (S3, T3, 1), # M3
            (S4, T4, 1), # M4
            (S5, T5, 1), # M5
            (S6, T6, 1), # M6
            (S7, T7, 1)  # M7
        ]
        
        # Coefficients for P[0][0], P[0][1], P[1][0], P[1][1]
        # P[0][0] = M1 - M4 + M5 - M7
        # P[0][1] = M3 + M7
        # P[1][0] = M2 + M4
        # P[1][1] = M1 - M2 + M3 + M6
        
        coeffs = [
            (1, 0, 0, -1),  # M1: P00=1, P01=0, P10=0, P11=-1? No.
            # M1 is in P00 (+1) and P11 (+1).
            # Let's map M_idx to coeffs (P00, P01, P10, P11)
        ]
        
        # M1: P00 +1, P11 +1
        # M2: P10 +1, P11 -1
        # M3: P01 +1, P11 +1
        # M4: P00 -1, P10 +1
        # M5: P00 +1
        # M6: P11 +1
        # M7: P00 -1, P01 +1
        
        m_coeffs = [
            (1, 0, 0, 1),  # M1
            (0, 0, 1, -1), # M2
            (0, 1, 0, 1),  # M3
            (-1, 0, 1, 0), # M4
            (1, 0, 0, 0),  # M5
            (0, 0, 0, 1),  # M6
            (-1, 1, 0, 0)  # M7
        ]
        
        for t_idx in range(7):
            x_list, y_list, _ = terms[t_idx]
            c00, c01, c10, c11 = m_coeffs[t_idx]
            
            if col >= rank: break
            
            # Fill U and V
            for (i_x, j_x, c_x) in x_list:
                U[i_x*m + j_x, col] = c_x
            for (j_y, k_y, c_y) in y_list:
                V[j_y*p + k_y, col] = c_y
            
            # Fill W
            # P[0][0] is C[0:2, 0:2]
            if c00 != 0:
                for i in range(2):
                    for k in range(2):
                        W[k*n + i, col] = c00
            # P[0][1] is C[0:2, 2:4]
            if c01 != 0:
                for i in range(2):
                    for k in range(2, 4):
                        W[k*n + i, col] = c01
            # P[1][0] is C[2:4, 0:2]
            if c10 != 0:
                for i in range(2, 4):
                    for k in range(2):
                        W[k*n + i, col] = c10
            # P[1][1] is C[2:4, 2:4]
            if c11 != 0:
                for i in range(2, 4):
                    for k in range(2, 4):
                        W[k*n + i, col] = c11
                        
            col += 1
            
    # Convert to lists of lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
