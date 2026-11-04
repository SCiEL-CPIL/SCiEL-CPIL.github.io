DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (32, rank)
    
    # We'll use rank 98 (2 * 49)
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the add_4x4x4_decomp function inside
    def add_4x4x4_decomp(k_offset, start_col):
        col = start_col
        
        def add_2x2x2_strassen(X_entries, Y_entries, col):
            for sx_idx in range(7):
                if sx_idx == 0:  # S1
                    X_combo = [(1, 0, 0), (1, 1, 1)]
                    Y_combo = [(1, 0, 0), (1, 1, 1)]
                    Z_coeff = [(0, 1), (1, 1)]
                elif sx_idx == 1:  # S2
                    X_combo = [(1, 1, 0), (1, 1, 1)]
                    Y_combo = [(1, 0, 0)]
                    Z_coeff = [(1, 1), (-1, 1)]
                elif sx_idx == 2:  # S3
                    X_combo = [(1, 0, 0)]
                    Y_combo = [(1, 0, 1), (-1, 1, 1)]
                    Z_coeff = [(0, 1), (1, 1)]
                elif sx_idx == 3:  # S4
                    X_combo = [(1, 1, 1)]
                    Y_combo = [(1, 1, 0), (-1, 0, 0)]
                    Z_coeff = [(0, 0), (1, 0)]
                elif sx_idx == 4:  # S5
                    X_combo = [(1, 0, 0), (1, 0, 1)]
                    Y_combo = [(1, 1, 1)]
                    Z_coeff = [(-1, 0), (1, 1)]
                elif sx_idx == 5:  # S6
                    X_combo = [(1, 1, 0), (-1, 0, 0)]
                    Y_combo = [(1, 0, 0), (1, 0, 1)]
                    Z_coeff = [(1, 1)]
                else:  # S7
                    X_combo = [(1, 0, 1), (-1, 1, 1)]
                    Y_combo = [(1, 1, 0), (1, 1, 1)]
                    Z_coeff = [(0, 0)]
                
                U_contrib = {}
                V_contrib = {}
                W_contrib = {}
                
                for sx_coeff, x_i, x_j in X_combo:
                    for a_coeff, a_row, a_col in X_entries[x_i*2+x_j]:
                        key = (a_row, a_col)
                        U_contrib[key] = U_contrib.get(key, 0) + sx_coeff * a_coeff
                
                for sx_coeff, y_i, y_j in Y_combo:
                    for b_coeff, b_row, b_col in Y_entries[y_i*2+y_j]:
                        key = (b_row, b_col)
                        V_contrib[key] = V_contrib.get(key, 0) + sx_coeff * b_coeff
                
                for c_coeff, c_row, c_col in Z_coeff:
                    key = (c_row, c_col)
                    W_contrib[key] = W_contrib.get(key, 0) + c_coeff
                
                for (a_row, a_col), coeff in U_contrib.items():
                    if coeff != 0:
                        U[a_row*4 + a_col][col] += coeff
                
                for (b_row, b_col), coeff in V_contrib.items():
                    if coeff != 0:
                        V[b_row*8 + (k_offset + b_col)][col] += coeff
                
                for (c_row, c_col), coeff in W_contrib.items():
                    if coeff != 0:
                        W[(k_offset + c_col)*4 + c_row][col] += coeff
                
                col += 1
            
            return col
        
        # M1 = (A11+A22) * (B11+B22)
        X_M1 = [
            [(1, 0, 0), (1, 2, 2)],
            [(1, 0, 1), (1, 2, 3)],
            [(1, 1, 0), (1, 3, 2)],
            [(1, 1, 1), (1, 3, 3)]
        ]
        Y_M1 = [
            [(1, 0, 0), (1, 2, 2)],
            [(1, 0, 1), (1, 2, 3)],
            [(1, 1, 0), (1, 3, 2)],
            [(1, 1, 1), (1, 3, 3)]
        ]
        col = add_2x2x2_strassen(X_M1, Y_M1, col)
        
        # M2 = (A21+A22) * B11
        X_M2 = [
            [(1, 2, 0), (1, 2, 2)],
            [(1, 2, 1), (1, 2, 3)],
            [(1, 3, 0), (1, 3, 2)],
            [(1, 3, 1), (1, 3, 3)]
        ]
        Y_M2 = [
            [(1, 0, 0)],
            [(1, 0, 1)],
            [(1, 1, 0)],
            [(1, 1, 1)]
        ]
        col = add_2x2x2_strassen(X_M2, Y_M2, col)
        
        # M3 = A11 * (B12-B22)
        X_M3 = [
            [(1, 0, 0)],
            [(1, 0, 1)],
            [(1, 1, 0)],
            [(1, 1, 1)]
        ]
        Y_M3 = [
            [(1, 0, 2), (-1, 2, 2)],
            [(1, 0, 3), (-1, 2, 3)],
            [(1, 1, 2), (-1, 3, 2)],
            [(1, 1, 3), (-1, 3, 3)]
        ]
        col = add_2x2x2_strassen(X_M3, Y_M3, col)
        
        # M4 = A22 * (B21-B11)
        X_M4 = [
            [(1, 2, 2)],
            [(1, 2, 3)],
            [(1, 3, 2)],
            [(1, 3, 3)]
        ]
        Y_M4 = [
            [(1, 2, 0), (-1, 0, 0)],
            [(1, 2, 1), (-1, 0, 1)],
            [(1, 3, 0), (-1, 1, 0)],
            [(1, 3, 1), (-1, 1, 1)]
        ]
        col = add_2x2x2_strassen(X_M4, Y_M4, col)
        
        # M5 = (A11+A12) * B22
        X_M5 = [
            [(1, 0, 0), (1, 0, 2)],
            [(1, 0, 1), (1, 0, 3)],
            [(1, 1, 0), (1, 1, 2)],
            [(1, 1, 1), (1, 1, 3)]
        ]
        Y_M5 = [
            [(1, 2, 2)],
            [(1, 2, 3)],
            [(1, 3, 2)],
            [(1, 3, 3)]
        ]
        col = add_2x2x2_strassen(X_M5, Y_M5, col)
        
        # M6 = (A21-A11) * (B11+B12)
        X_M6 = [
            [(1, 2, 0), (-1, 0, 0)],
            [(1, 2, 1), (-1, 0, 1)],
            [(1, 3, 0), (-1, 1, 0)],
            [(1, 3, 1), (-1, 1, 1)]
        ]
        Y_M6 = [
            [(1, 0, 0), (1, 0, 2)],
            [(1, 0, 1), (1, 0, 3)],
            [(1, 1, 0), (1, 1, 2)],
            [(1, 1, 1), (1, 1, 3)]
        ]
        col = add_2x2x2_strassen(X_M6, Y_M6, col)
        
        # M7 = (A12-A22) * (B21+B22)
        X_M7 = [
            [(1, 0, 2), (-1, 2, 2)],
            [(1, 0, 3), (-1, 2, 3)],
            [(1, 1, 2), (-1, 3, 2)],
            [(1, 1, 3), (-1, 3, 3)]
        ]
        Y_M7 = [
            [(1, 2, 0), (1, 2, 2)],
            [(1, 2, 1), (1, 2, 3)],
            [(1, 3, 0), (1, 3, 2)],
            [(1, 3, 1), (1, 3, 3)]
        ]
        col = add_2x2x2_strassen(X_M7, Y_M7, col)
        
        return col
    
    # Left half: k_offset=0
    col = add_4x4x4_decomp(0, 0)
    # Right half: k_offset=4
    col = add_4x4x4_decomp(4, col)
    
    return {"U": U, "V": V, "W": W}
