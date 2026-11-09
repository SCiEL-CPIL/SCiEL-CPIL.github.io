def construct():
    n, m, p = 2, 4, 4
    # Rank 28 decomposition using block-Strassen
    # A is 2x4, B is 4x4, C is 2x4
    # Split A into A1 (2x2, cols 0,1) and A2 (2x2, cols 2,3)
    # Split B into 2x2 blocks:
    #   B = [[B11, B12],
    #        [B21, B22]]
    # where B11 is rows 0,1 cols 0,1
    #       B12 is rows 0,1 cols 2,3
    #       B21 is rows 2,3 cols 0,1
    #       B22 is rows 2,3 cols 2,3
    #
    # C = [A1, A2] * [[B11, B12],
    #                  [B21, B22]]
    #   = [A1*B11 + A2*B21,  A1*B12 + A2*B22]
    #
    # For each 2x2 * 2x2 product, use Strassen (7 multiplications)
    # Total: 4 products * 7 = 28 multiplications

    rank = 28
    # U: 8 x rank (indices: i*m + j, i in 0..1, j in 0..3)
    # V: 16 x rank (indices: j*p + k, j in 0..3, k in 0..3)
    # W: 8 x rank (indices: k*n + i, k in 0..3, i in 0..1)
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen factors for 2x2 * 2x2
    # Given matrices X (2x2) and Y (2x2), compute Z = X*Y
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z = [[z00, z01], [z10, z11]]
    #
    # Strassen:
    # M1 = (x00+x11)*(y00+y11)
    # M2 = (x10+x11)*y00
    # M3 = x00*(y01-y11)
    # M4 = x11*(y10-y00)
    # M5 = (x00+x01)*y11
    # M6 = (x10-x00)*(y00+y10)
    # M7 = (x01-x11)*(y01+y11)
    #
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    def strassen_factors(x_idx, y_idx):
        """
        Return list of (x_coeff, y_coeff) pairs for each Strassen multiplication.
        x_idx: list of 4 indices for X elements [x00, x01, x10, x11]
        y_idx: list of 4 indices for Y elements [y00, y01, y10, y11]
        Returns list of 7 tuples, each tuple is a list of (x_index, coeff) and (y_index, coeff)
        """
        # Each multiplication involves a sum of X elements times a sum of Y elements
        # M1: (x00+x11)*(y00+y11)
        m1_x = [(x_idx[0], 1), (x_idx[2], 1)]
        m1_y = [(y_idx[0], 1), (y_idx[3], 1)]
        
        # M2: (x10+x11)*y00
        m2_x = [(x_idx[2], 1), (x_idx[3], 1)]
        m2_y = [(y_idx[0], 1)]
        
        # M3: x00*(y01-y11)
        m3_x = [(x_idx[0], 1)]
        m3_y = [(y_idx[1], 1), (y_idx[3], -1)]
        
        # M4: x11*(y10-y00)
        m4_x = [(x_idx[3], 1)]
        m4_y = [(y_idx[2], 1), (y_idx[0], -1)]
        
        # M5: (x00+x01)*y11
        m5_x = [(x_idx[0], 1), (x_idx[1], 1)]
        m5_y = [(y_idx[3], 1)]
        
        # M6: (x10-x00)*(y00+y10)
        m6_x = [(x_idx[2], 1), (x_idx[0], -1)]
        m6_y = [(y_idx[0], 1), (y_idx[2], 1)]
        
        # M7: (x01-x11)*(y01+y11)
        m7_x = [(x_idx[1], 1), (x_idx[3], -1)]
        m7_y = [(y_idx[1], 1), (y_idx[3], 1)]
        
        return [
            (m1_x, m1_y),
            (m2_x, m2_y),
            (m3_x, m3_y),
            (m4_x, m4_y),
            (m5_x, m5_y),
            (m6_x, m6_y),
            (m7_x, m7_y)
        ]
    
    def strassen_output_coeffs():
        """
        Return the coefficients for each output element in terms of M1..M7
        z00 = M1 + M4 - M5 + M7
        z01 = M3 + M5
        z10 = M2 + M4
        z11 = M1 - M2 + M3 + M6
        """
        return {
            0: [(0, 1), (3, 1), (4, -1), (6, 1)],   # z00
            1: [(2, 1), (4, 1)],                      # z01
            2: [(1, 1), (3, 1)],                      # z10
            3: [(0, 1), (1, -1), (2, 1), (5, 1)]      # z11
        }
    
    output_coeffs = strassen_output_coeffs()
    
    # We have 4 Strassen decompositions:
    # 1. A1 * B11 -> contributes to C_left (cols 0,1)
    # 2. A2 * B21 -> contributes to C_left (cols 0,1)
    # 3. A1 * B12 -> contributes to C_right (cols 2,3)
    # 4. A2 * B22 -> contributes to C_right (cols 2,3)
    
    # A1: A[0,0], A[0,1], A[1,0], A[1,1] -> indices in U: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # A2: A[0,2], A[0,3], A[1,2], A[1,3] -> indices in U: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    
    # B11: B[0,0], B[0,1], B[1,0], B[1,1] -> indices in V: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # B12: B[0,2], B[0,3], B[1,2], B[1,3] -> indices in V: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # B21: B[2,0], B[2,1], B[3,0], B[3,1] -> indices in V: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # B22: B[2,2], B[2,3], B[3,2], B[3,3] -> indices in V: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    
    # C outputs:
    # C_left: C[0,0], C[0,1], C[1,0], C[1,1] -> indices in W: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3
    # C_right: C[0,2], C[0,3], C[1,2], C[1,3] -> indices in W: 2*2+0=4, 2*2+1=5, 3*2+0=6, 3*2+1=7
    
    # For each of the 4 Strassen decompositions, assign 7 columns
    # Decomposition 1: A1 * B11, columns 0-6
    # Decomposition 2: A2 * B21, columns 7-13
    # Decomposition 3: A1 * B12, columns 14-20
    # Decomposition 4: A2 * B22, columns 21-27
    
    decompositions = [
        {
            'x_idx': [0, 1, 4, 5],  # A1
            'y_idx': [0, 1, 4, 5],  # B11
            'col_offset': 0,
            # Output: z00 -> C[0,0]=W[0], z01 -> C[0,1]=W[1], z10 -> C[1,0]=W[2], z11 -> C[1,1]=W[3]
            'out_map': [0, 1, 2, 3]
        },
        {
            'x_idx': [2, 3, 6, 7],  # A2
            'y_idx': [8, 9, 12, 13],  # B21
            'col_offset': 7,
            # Output: z00 -> C[0,0]=W[0], z01 -> C[0,1]=W[1], z10 -> C[1,0]=W[2], z11 -> C[1,1]=W[3]
            'out_map': [0, 1, 2, 3]
        },
        {
            'x_idx': [0, 1, 4, 5],  # A1
            'y_idx': [2, 3, 6, 7],  # B12
            'col_offset': 14,
            # Output: z00 -> C[0,2]=W[4], z01 -> C[0,3]=W[5], z10 -> C[1,2]=W[6], z11 -> C[1,3]=W[7]
            'out_map': [4, 5, 6, 7]
        },
        {
            'x_idx': [2, 3, 6, 7],  # A2
            'y_idx': [10, 11, 14, 15],  # B22
            'col_offset': 21,
            # Output: z00 -> C[0,2]=W[4], z01 -> C[0,3]=W[5], z10 -> C[1,2]=W[6], z11 -> C[1,3]=W[7]
            'out_map': [4, 5, 6, 7]
        }
    ]
    
    for dec in decompositions:
        x_idx = dec['x_idx']
        y_idx = dec['y_idx']
        col_offset = dec['col_offset']
        out_map = dec['out_map']
        
        factors = strassen_factors(x_idx, y_idx)
        
        for m_idx in range(7):
            col = col_offset + m_idx
            x_terms, y_terms = factors[m_idx]
            
            for x_i, coeff in x_terms:
                U[x_i][col] += coeff
            for y_i, coeff in y_terms:
                V[y_i][col] += coeff
            
            # For each output element, add its contribution
            for z_idx in range(4):
                w_idx = out_map[z_idx]
                for m_j, coeff in output_coeffs[z_idx]:
                    if m_j == m_idx:
                        W[w_idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}
