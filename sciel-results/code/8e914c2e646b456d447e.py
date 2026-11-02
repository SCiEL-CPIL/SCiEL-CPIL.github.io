import time

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen for 2x2x2 blocks
    # 4 blocks of 2x2x2
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Local indices:
    # U: i*2+j (0..3)
    # V: j*2+k (0..3)
    # W: k*2+i (0..3)
    
    # Term 1: (a+d)(e+h)
    u1 = [1, 0, 1, 0] # a, b, c, d -> a=1, d=1
    v1 = [1, 1, 1, 1] # e, f, g, h -> e=1, f=1, g=1, h=1
    w1 = [1, 1, 1, 1] # C00, C01, C10, C11
    
    # Term 2: (c+d)e
    u2 = [0, 0, 1, 1] # c=1, d=1
    v2 = [1, 0, 0, 0] # e=1
    w2 = [0, 0, 1, -1] # C10+=1, C11-=1
    
    # Term 3: a(f-h)
    u3 = [1, 0, 0, 0] # a=1
    v3 = [0, 1, 0, -1] # f=1, h=-1
    w3 = [0, 1, 0, 1] # C01+=1, C11+=1
    
    # Term 4: d(g-e)
    u4 = [0, 0, 0, 1] # d=1
    v4 = [-1, 0, 1, 0] # e=-1, g=1
    w4 = [1, 0, 1, 0] # C00+=1, C10+=1
    
    # Term 5: (a+b)h
    u5 = [1, 1, 0, 0] # a=1, b=1
    v5 = [0, 0, 0, 1] # h=1
    w5 = [-1, 1, 0, 0] # C00-=1, C01+=1
    
    # Term 6: (c-a)(e+f)
    u6 = [-1, 0, 1, 0] # a=-1, c=1
    v6 = [1, 1, 0, 0] # e=1, f=1
    w6 = [0, 0, 0, 1] # C11+=1
    
    # Term 7: (b-d)(g+h)
    u7 = [0, 1, 0, -1] # b=1, d=-1
    v7 = [0, 0, 1, 1] # g=1, h=1
    w7 = [1, 0, 0, 0] # C00+=1
    
    terms = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    for b in range(num_blocks):
        for s in range(7):
            c = b * 7 + s
            u_loc, v_loc, w_loc = terms[s]
            
            # Fill U (same for all blocks)
            for i_loc in range(2):
                for j_loc in range(2):
                    u_idx = i_loc * 2 + j_loc
                    U[u_idx][c] = u_loc[u_idx]
            
            # Fill V
            for j_loc in range(2):
                for k_loc in range(2):
                    v_idx_loc = j_loc * 2 + k_loc
                    k_glob = 2 * b + k_loc
                    v_idx_glob = j_loc * p + k_glob
                    V[v_idx_glob][c] = v_loc[v_idx_loc]
            
            # Fill W
            for k_loc in range(2):
                for i_loc in range(2):
                    w_idx_loc = k_loc * 2 + i_loc
                    k_glob = 2 * b + k_loc
                    w_idx_glob = k_glob * n + i_loc
                    W[w_idx_glob][c] = w_loc[w_idx_loc]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
