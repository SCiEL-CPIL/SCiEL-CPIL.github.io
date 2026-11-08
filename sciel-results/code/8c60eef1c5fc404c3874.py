"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    def strassen_2x2_terms():
        terms = []
        terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
        terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1}))
        terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1}))
        terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}))
        terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}))
        terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
        terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}))
        return terms
    
    def build_4x4x4_strassen():
        s2_terms = strassen_2x2_terms()
        rank_4x4 = 49
        U = [[0]*rank_4x4 for _ in range(16)]
        V = [[0]*rank_4x4 for _ in range(16)]
        W = [[0]*rank_4x4 for _ in range(16)]
        
        col = 0
        for (u2, v2, w2) in s2_terms:
            for (u2r, u2c), u2v in u2.items():
                for (v2r, v2c), v2v in v2.items():
                    for (w2r, w2c), w2v in w2.items():
                        for li in range(2):
                            for lj in range(2):
                                for lk in range(2):
                                    a_row = u2r*2 + li
                                    a_col = u2c*2 + lj
                                    b_row = v2r*2 + lj
                                    b_col = v2c*2 + lk
                                    c_i = w2c*2 + li
                                    c_k = w2r*2 + lk
                                    w_idx = c_k * 4 + c_i
                                    
                                    U[a_row*4 + a_col][col] += u2v
                                    V[b_row*4 + b_col][col] += v2v
                                    W[w_idx][col] += w2v
                        col += 1
                        if col >= rank_4x4:
                            break
                    if col >= rank_4x4:
                        break
                if col >= rank_4x4:
                    break
            if col >= rank_4x4:
                break
        
        return U, V, W
    
    U4, V4, W4 = build_4x4x4_strassen()
    
    rank_total = 98
    U = [[0]*rank_total for _ in range(16)]
    V = [[0]*rank_total for _ in range(32)]
    W = [[0]*rank_total for _ in range(32)]
    
    for r in range(16):
        for c in range(49):
            U[r][c] = U4[r][c]
            U[r][c+49] = U4[r][c]
            V[r][c] = V4[r][c]
            V[r+16][c+49] = V4[r][c]
            W[r][c] = W4[r][c]
            W[r+16][c+49] = W4[r][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
