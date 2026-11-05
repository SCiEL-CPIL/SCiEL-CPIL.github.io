DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, [(0,0,1), (1,1,1)]),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, [(1,0,1), (1,1,-1)]),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, [(0,1,1), (1,1,1)]),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, [(0,0,1), (1,0,1)]),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, [(0,0,-1), (0,1,1)]),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, [(1,1,1)]),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, [(0,0,1)]),
    ]
    
    col = 0
    for a_coeffs, b_coeffs, c_contribs in strassen_terms:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for (abr, abc), a_c in a_coeffs.items():
                        u_idx = (abr * 3 + i) * 6 + (abc * 3 + j)
                        U[u_idx][col] = a_c
                    for (bbr, bbc), b_c in b_coeffs.items():
                        v_idx = (bbr * 3 + j) * 6 + (bbc * 3 + k)
                        V[v_idx][col] = b_c
                    for (ccr, ccc, s) in c_contribs:
                        w_idx = (ccc * 3 + k) * 6 + (ccr * 3 + i)
                        W[w_idx][col] = s
                    col += 1
    
    return {"U": U, "V": V, "W": W}
