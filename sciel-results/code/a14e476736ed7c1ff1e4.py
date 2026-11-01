"""Program for matrix multiplication 2,2,10 using block-diagonal Strassen decomposition."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    
    # Strassen terms: (u_list, v_list, w_list)
    # u_list: [(i, j, val), ...]
    # v_list: [(j, k, val), ...]
    # w_list: [(k, i, val), ...]
    terms = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (1,0,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    rank = num_blocks * 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r in range(num_blocks):
        for t, (u_list, v_list, w_list) in enumerate(terms):
            col = r * 7 + t
            for i, j, val in u_list:
                U[i*m + j][col] = val
            for j, k, val in v_list:
                V[j*p + (2*r + k)][col] = val
            for k, i, val in w_list:
                W[(2*r + k)*n + i][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
