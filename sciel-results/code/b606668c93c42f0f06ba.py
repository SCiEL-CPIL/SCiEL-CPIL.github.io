    def construct():
        n, m, p = 2, 2, 14
        rank = 49
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen coefficients for U (a,b,c,d)
        # Term 0: M1 = (a+d)(e+h) -> U: [1, 0, 0, 1]
        # Term 1: M2 = (c+d)e     -> U: [0, 0, 1, 1]
        # Term 2: M3 = a(f-h)     -> U: [1, 0, 0, 0]
        # Term 3: M4 = d(g-f)     -> U: [0, 0, 0, 1]
        # Term 4: M5 = (a+b)h     -> U: [1, 1, 0, 0]
        # Term 5: M6 = (c-a)(e+g) -> U: [-1, 0, 1, 0]
        # Term 6: M7 = (b-d)(f+h) -> U: [0, 1, 0, -1]
        
        U_coeffs = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        
        # V indices relative to block l
        # e: 2l, f: 2l+1, g: 14+2l, h: 14+2l+1
        # Term 0: e, h
        # Term 1: e
        # Term 2: f, h
        # Term 3: g, f
        # Term 4: h
        # Term 5: e, g
        # Term 6: f, h
        
        # W indices relative to block l
        # C00: 4l, C10: 4l+1, C01: 4l+2, C11: 4l+3
        # Term 0: C00(+1), C11(+1)
        # Term 1: C10(+1), C11(-1)
        # Term 2: C01(+1), C11(+1)
        # Term 3: C00(+1), C10(+1)
        # Term 4: C00(-1), C01(+1)
        # Term 5: C11(+1)
        # Term 6: C00(+1)
        
        for l in range(7):
            for t in range(7):
                c = 7 * l + t
                
                # U
                for i in range(4):
                    U[i][c] = U_coeffs[t][i]
                    
                # V
                # Determine which B elements are involved
                # e: 2l, f: 2l+1, g: 14+2l, h: 14+2l+1
                if t == 0: # e, h
                    V[2*l][c] = 1
                    V[14+2*l+1][c] = 1
                elif t == 1: # e
                    V[2*l][c] = 1
                elif t == 2: # f, h
                    V[2*l+1][c] = 1
                    V[14+2*l+1][c] = -1
                elif t == 3: # g, f
                    V[14+2*l][c] = 1
                    V[2*l+1][c] = -1
                elif t == 4: # h
                    V[14+2*l+1][c] = 1
                elif t == 5: # e, g
                    V[2*l][c] = 1
                    V[14+2*l][c] = 1
                elif t == 6: # f, h
                    V[2*l+1][c] = 1
                    V[14+2*l+1][c] = 1
                    
                # W
                if t == 0: # C00, C11
                    W[4*l][c] = 1
                    W[4*l+3][c] = 1
                elif t == 1: # C10, C11
                    W[4*l+1][c] = 1
                    W[4*l+3][c] = -1
                elif t == 2: # C01, C11
                    W[4*l+2][c] = 1
                    W[4*l+3][c] = 1
                elif t == 3: # C00, C10
                    W[4*l][c] = 1
                    W[4*l+1][c] = 1
                elif t == 4: # C00, C01
                    W[4*l][c] = -1
                    W[4*l+2][c] = 1
                elif t == 5: # C11
                    W[4*l+3][c] = 1
                elif t == 6: # C00
                    W[4*l][c] = 1
                    
        return {"U": U, "V": V, "W": W}
    