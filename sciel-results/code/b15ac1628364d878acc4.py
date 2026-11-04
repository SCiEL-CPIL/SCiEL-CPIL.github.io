# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen tensors
    S = []
    S.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    S.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    S.append(([1,0,0,0], [0,0,1,-1], [0,1,0,1]))
    S.append(([0,0,0,1], [0,1,-1,0], [-1,1,0,0]))
    S.append(([1,1,0,0], [0,0,0,1], [1,0,1,0]))
    S.append(([0,0,1,-1], [1,0,1,0], [0,0,1,0]))
    S.append(([0,1,0,-1], [0,1,0,1], [-1,0,0,1]))
    
    # Top-level Strassen definitions
    # x: 2x2 coeffs for A blocks (0,0), (0,1), (1,0), (1,1)
    # y: 2x2 coeffs for B blocks
    # c_contribs: list of (block_row, block_col, sign)
    
    T = []
    # M1
    T.append(([1,0,0,1], [1,0,0,1], [(0,0,1), (1,1,1)]))
    # M2
    T.append(([0,0,1,1], [1,0,0,0], [(1,0,1)]))
    # M3
    T.append(([1,0,0,0], [0,0,1,-1], [(0,1,1), (1,1,1)]))
    # M4
    T.append(([0,0,0,1], [0,1,-1,0], [(0,0,-1), (0,1,1)]))
    # M5
    T.append(([1,1,0,0], [0,0,0,1], [(0,0,1), (1,0,1)]))
    # M6
    T.append(([0,0,1,-1], [1,0,1,0], [(1,0,1)]))
    # M7
    T.append(([0,1,0,-1], [0,1,0,1], [(0,0,-1), (1,1,1)]))
    
    col_idx = 0
    for i in range(7):
        x = T[i][0]
        y = T[i][1]
        c_contribs = T[i][2]
        
        for s in range(7):
            Us, Vs, Ws = S[s]
            # Expand
            for r in range(2):
                for c in range(2):
                    # U coeff
                    for a in range(2):
                        for b in range(2):
                            coeff = Us[r*2+c] * x[a*2+b]
                            if coeff != 0:
                                ui = (2*a + r) * 4 + (2*b + c)
                                U[ui][col_idx] += coeff
                                
                    # V coeff
                    for c_block in range(2):
                        for d_block in range(2):
                            coeff = Vs[r*2+c] * y[c_block*2+d_block]
                            if coeff != 0:
                                vi = (2*c_block + r) * 7 + (2*d_block + c)
                                V[vi][col_idx] += coeff
                                
                    # W coeff
                    for cr in range(2):
                        for cc in range(2):
                            w_val = Ws[r*2+c]
                            if w_val != 0:
                                for cb_r, cb_c, sign in c_contribs:
                                    wi = (2*cb_c + c) * 4 + (2*cb_r + r)
                                    W[wi][col_idx] += sign * w_val
            col_idx += 1
            
    # Schoolbook for 4x4 x 4x3
    for i in range(4):
        for j in range(4):
            for k in range(3):
                K = k + 4
                U[i*4 + j][col_idx] = 1
                V[j*7 + K][col_idx] = 1
                W[K*4 + i][col_idx] = 1
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
