import time

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 groups of Strassen
    # Each group g has X_g and Y_g
    # X_g and Y_g are 2x2 matrices of linear forms
    # We represent a linear form as a list of 16 coefficients
    
    # A indices: i*4 + j
    # B indices: j*4 + k
    # C indices: k*4 + i
    
    # Helper to get index for A
    def a_idx(i, j):
        return i * 4 + j
        
    # Helper to get index for B
    def b_idx(j, k):
        return j * 4 + k
        
    # Helper to get index for C
    def c_idx(k, i):
        return k * 4 + i
        
    # Define the 7 block combinations
    # X_g[u, v] is a list of 16 coeffs for A
    # Y_g[u, v] is a list of 16 coeffs for B
    
    X = [[None for _ in range(2)] for _ in range(7)]
    Y = [[None for _ in range(2)] for _ in range(7)]
    
    # Initialize with zeros
    for g in range(7):
        for u in range(2):
            for v in range(2):
                X[g][u][v] = [0] * 16
                Y[g][u][v] = [0] * 16
                
    # g=0: X = A00 + A11, Y = B00 + B11
    # A00 is block (0,0), A11 is block (1,1)
    # Block (i, j) of A has elements A[2i+u, 2j+v]
    for u in range(2):
        for v in range(2):
            X[0][u][v][a_idx(0, 0)] = 1 # A00
            X[0][u][v][a_idx(1, 1)] = 1 # A11
            Y[0][u][v][b_idx(0, 0)] = 1 # B00
            Y[0][u][v][b_idx(1, 1)] = 1 # B11
            
    # g=1: X = A10 + A11, Y = B00
    for u in range(2):
        for v in range(2):
            X[1][u][v][a_idx(1, 0)] = 1 # A10
            X[1][u][v][a_idx(1, 1)] = 1 # A11
            Y[1][u][v][b_idx(0, 0)] = 1 # B00
            
    # g=2: X = A00, Y = B01 - B11
    for u in range(2):
        for v in range(2):
            X[2][u][v][a_idx(0, 0)] = 1 # A00
            Y[2][u][v][b_idx(0, 1)] = 1 # B01
            Y[2][u][v][b_idx(1, 1)] = -1 # B11
            
    # g=3: X = A11, Y = B10 - B00
    for u in range(2):
        for v in range(2):
            X[3][u][v][a_idx(1, 1)] = 1 # A11
            Y[3][u][v][b_idx(1, 0)] = 1 # B10
            Y[3][u][v][b_idx(0, 0)] = -1 # B00
            
    # g=4: X = A00 + A01, Y = B11
    for u in range(2):
        for v in range(2):
            X[4][u][v][a_idx(0, 0)] = 1 # A00
            X[4][u][v][a_idx(0, 1)] = 1 # A01
            Y[4][u][v][b_idx(1, 1)] = 1 # B11
            
    # g=5: X = A10 - A00, Y = B00 + B01
    for u in range(2):
        for v in range(2):
            X[5][u][v][a_idx(1, 0)] = 1 # A10
            X[5][u][v][a_idx(0, 0)] = -1 # A00
            Y[5][u][v][b_idx(0, 0)] = 1 # B00
            Y[5][u][v][b_idx(0, 1)] = 1 # B01
            
    # g=6: X = A01 - A11, Y = B10 + B11
    for u in range(2):
        for v in range(2):
            X[6][u][v][a_idx(0, 1)] = 1 # A01
            X[6][u][v][a_idx(1, 1)] = -1 # A11
            Y[6][u][v][b_idx(1, 0)] = 1 # B10
            Y[6][u][v][b_idx(1, 1)] = 1 # B11
            
    # Now, for each group g, compute the 7 scalar multiplications
    # m_{g, s} = u_{g, s} * v_{g, s}
    
    # Strassen formulas for 2x2 matrices X, Y:
    # m0 = (x00 + x11)(y00 + y11)
    # m1 = (x10 + x11)y00
    # m2 = x00(y01 - y11)
    # m3 = x11(y10 - y00)
    # m4 = (x00 + x01)y11
    # m5 = (x10 - x00)(y00 + y01)
    # m6 = (x01 - x11)(y10 + y11)
    
    for g in range(7):
        # m0
        u0 = [X[g][0][0][i] + X[g][1][1][i] for i in range(16)]
        v0 = [Y[g][0][0][i] + Y[g][1][1][i] for i in range(16)]
        col = 7 * g + 0
        for i in range(16):
            U[i][col] = u0[i]
            V[i][col] = v0[i]
            
        # m1
        u1 = [X[g][1][0][i] + X[g][1][1][i] for i in range(16)]
        v1 = [Y[g][0][0][i] for i in range(16)]
        col = 7 * g + 1
        for i in range(16):
            U[i][col] = u1[i]
            V[i][col] = v1[i]
            
        # m2
        u2 = [X[g][0][0][i] for i in range(16)]
        v2 = [Y[g][0][1][i] - Y[g][1][1][i] for i in range(16)]
        col = 7 * g + 2
        for i in range(16):
            U[i][col] = u2[i]
            V[i][col] = v2[i]
            
        # m3
        u3 = [X[g][1][1][i] for i in range(16)]
        v3 = [Y[g][1][0][i] - Y[g][0][0][i] for i in range(16)]
        col = 7 * g + 3
        for i in range(16):
            U[i][col] = u3[i]
            V[i][col] = v3[i]
            
        # m4
        u4 = [X[g][0][0][i] + X[g][0][1][i] for i in range(16)]
        v4 = [Y[g][1][1][i] for i in range(16)]
        col = 7 * g + 4
        for i in range(16):
            U[i][col] = u4[i]
            V[i][col] = v4[i]
            
        # m5
        u5 = [X[g][1][0][i] - X[g][0][0][i] for i in range(16)]
        v5 = [Y[g][0][0][i] + Y[g][0][1][i] for i in range(16)]
        col = 7 * g + 5
        for i in range(16):
            U[i][col] = u5[i]
            V[i][col] = v5[i]
            
        # m6
        u6 = [X[g][0][1][i] - X[g][1][1][i] for i in range(16)]
        v6 = [Y[g][1][0][i] + Y[g][1][1][i] for i in range(16)]
        col = 7 * g + 6
        for i in range(16):
            U[i][col] = u6[i]
            V[i][col] = v6[i]
            
    # Now construct W
    # C_{2i+u, 2j+v} = sum_{g} w_{g, i, j} (M_g)_{u, v}
    # (M_g)_{u, v} is a linear combination of m_{g, s}
    
    # Strassen reconstruction:
    # Z00 = m0 + m3 - m4 + m6
    # Z01 = m2 + m4
    # Z10 = m1 + m3
    # Z11 = m0 - m1 + m2 + m5
    
    # Block C_ij corresponds to i, j in {0, 1}
    # C_00 = M0 + M3 - M4 + M6
    # C_01 = M2 + M4
    # C_10 = M1 + M3
    # C_11 = M0 - M1 + M2 + M5
    
    # For each block position (i, j) in C, and each element (u, v) in the block:
    # C[2i+u, 2j+v] = sum_{g} coeff_{g, i, j, u, v} * (M_g)_{u, v}
    
    # Let's define the coefficients for (M_g)_{u, v} in terms of m_{g, s}
    # m_{g, s} is at column 7*g + s
    
    # For a fixed g, u, v, the contribution to (M_g)_{u, v} from m_{g, s} is:
    # if u=0, v=0: m0 + m3 - m4 + m6
    # if u=0, v=1: m2 + m4
    # if u=1, v=0: m1 + m3
    # if u=1, v=1: m0 - m1 + m2 + m5
    
    # So, for each g, u, v, we have a vector of 7 coefficients for m_{g, 0} ... m_{g, 6}
    
    # Now, for each block position (i, j) in C, we have a linear combination of M_g
    # C_00: M0 + M3 - M4 + M6
    # C_01: M2 + M4
    # C_10: M1 + M3
    # C_11: M0 - M1 + M2 + M5
    
    # So, for each block position (i, j), we have a vector of 7 coefficients for M_0 ... M_6
    
    # Let's combine these.
    # For each block position (i, j) and each element (u, v) in the block:
    # C[2i+u, 2j+v] = sum_{g} block_coeff_{g, i, j} * sum_{s} m_coeff_{s, u, v} * m_{g, s}
    
    # block_coeff_{g, i, j} is 1, -1, or 0
    # m_coeff_{s, u, v} is 1, -1, or 0
    
    # We will fill W[k*n+i, col] where col = 7*g + s
    # W index is k*4 + i
    
    for i in range(2): # block row
        for j in range(2): # block col
            for u in range(2): # element row in block
                for v in range(2): # element col in block
                    # C index
                    c_i = 2 * i + u
                    c_k = 2 * j + v
                    w_idx = c_k * 4 + c_i
                    
                    for g in range(7):
                        # Determine block_coeff_{g, i, j}
                        bc = 0
                        if i == 0 and j == 0:
                            if g == 0: bc = 1
                            elif g == 3: bc = 1
                            elif g == 4: bc = -1
                            elif g == 6: bc = 1
                        elif i == 0 and j == 1:
                            if g == 2: bc = 1
                            elif g == 4: bc = 1
                        elif i == 1 and j == 0:
                            if g == 1: bc = 1
                            elif g == 3: bc = 1
                        elif i == 1 and j == 1:
                            if g == 0: bc = 1
                            elif g == 1: bc = -1
                            elif g == 2: bc = 1
                            elif g == 5: bc = 1
                            
                        if bc == 0:
                            continue
                            
                        # Determine m_coeff_{s, u, v}
                        for s in range(7):
                            mc = 0
                            if u == 0 and v == 0:
                                if s == 0: mc = 1
                                elif s == 3: mc = 1
                                elif s == 4: mc = -1
                                elif s == 6: mc = 1
                            elif u == 0 and v == 1:
                                if s == 2: mc = 1
                                elif s == 4: mc = 1
                            elif u == 1 and v == 0:
                                if s == 1: mc = 1
                                elif s == 3: mc = 1
                            elif u == 1 and v == 1:
                                if s == 0: mc = 1
                                elif s == 1: mc = -1
                                elif s == 2: mc = 1
                                elif s == 5: mc = 1
                                
                            if mc == 0:
                                continue
                                
                            col = 7 * g + s
                            W[w_idx][col] += bc * mc
                            
    return {"U": U, "V": V, "W": W}
