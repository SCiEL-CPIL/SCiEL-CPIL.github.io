    def construct():
        n, m, p = 2, 4, 8
        rank = 56
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen coefficients
        # u_s[i, j0]
        u_coeffs = [
            [[1, 0], [0, 1]], # s0
            [[0, 0], [1, 1]], # s1
            [[1, 0], [0, 0]], # s2
            [[0, 0], [0, 1]], # s3
            [[1, 1], [0, 0]], # s4
            [[-1, 0], [1, 0]],# s5
            [[0, 1], [0, -1]] # s6
        ]
        
        # v_s[j0, k0]
        v_coeffs = [
            [[1, 0], [0, 1]], # s0
            [[1, 0], [0, 0]], # s1
            [[0, -1], [1, 0]],# s2 -> q10 - q11? No. q10 is j0=1, k0=0. q11 is j0=1, k0=1.
            # s2: p00(q10-q11). v[1,0]=1, v[1,1]=-1.
            [[0, 0], [1, -1]],
            [[-1, 1], [0, 0]],# s3: p11(q01-q00). v[0,1]=1, v[0,0]=-1.
            [[0, 0], [0, 1]], # s4
            [[1, 1], [0, 0]], # s5
            [[0, 0], [1, 1]]  # s6
        ]
        
        # w_s[i, k0]
        w_coeffs = [
            [1, 0, 0, 1, -1, 0, 1], # M00
            [0, 1, 1, 0, 0, -1, 1], # M01
            [0, 1, 1, 1, -1, 0, 0], # M10
            [1, 0, 0, 1, 0, 1, -1]  # M11
        ]
        
        for x in range(2):
            for k1 in range(4):
                for s in range(7):
                    c = (x * 4 + k1) * 7 + s
                    
                    # Fill U
                    for i in range(2):
                        for j0 in range(2):
                            val = u_coeffs[s][i][j0]
                            if val != 0:
                                U[i * 4 + (2 * x + j0)][c] = val
                                
                    # Fill V
                    for j0 in range(2):
                        for k0 in range(2):
                            val = v_coeffs[s][j0][k0]
                            if val != 0:
                                V[(2 * x + j0) * 8 + (2 * k1 + k0)][c] = val
                                
                    # Fill W
                    for i in range(2):
                        for k0 in range(2):
                            val = w_coeffs[i][s]
                            if val != 0:
                                W[(2 * k1 + k0) * 2 + i][c] = val
                                
        return {"U": U, "V": V, "W": W}
    