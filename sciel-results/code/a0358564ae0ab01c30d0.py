def construct():
    n, m, p = 2, 4, 8
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    S = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)],           [(1,0,1), (1,1,-1)]),
        ([(0,0,1)],           [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)],           [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)],  [(1,1,1)],           [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)],  [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)],  [(0,0,1)]),
    ]
    
    idx = 0
    for out_blk in range(4): # k
        for mid_blk in range(2): # j
            for u_terms, v_terms, w_terms in S:
                for (r, s, w) in u_terms:
                    u_row = r * m + (mid_blk * 2 + s)
                    U[u_row][idx] += w
                for (s, t, w) in v_terms:
                    v_row = (mid_blk * 2 + s) * p + (out_blk * 2 + t)
                    V[v_row][idx] += w
                for (r, t, w) in w_terms:
                    w_row = (out_blk * 2 + t) * n + r
                    W[w_row][idx] += w
                idx += 1
                
    return {"U": U, "V": V, "W": W}
