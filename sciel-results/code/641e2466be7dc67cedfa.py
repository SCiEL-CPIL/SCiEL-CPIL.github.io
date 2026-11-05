DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    bs = 3
    
    a_factors = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)],
    ]
    
    b_factors = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)],
    ]
    
    c_recon = [[None]*2 for _ in range(2)]
    c_recon[0][0] = {0:1, 3:1, 4:-1, 6:1}
    c_recon[0][1] = {2:1, 4:1}
    c_recon[1][0] = {1:1, 3:1}
    c_recon[1][1] = {0:1, 1:-1, 2:1, 5:1}
    
    rank = 7 * bs * bs * bs
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for s in range(7):
        for r in range(bs):
            for c in range(bs):
                for d in range(bs):
                    for (br, bc, coeff) in a_factors[s]:
                        U[(br*bs + r) * m + (bc*bs + d)][col] = coeff
                    for (br2, bc2, coeff) in b_factors[s]:
                        V[(br2*bs + d) * p + (bc2*bs + c)][col] = coeff
                    for br_c in range(2):
                        for bc_c in range(2):
                            if s in c_recon[br_c][bc_c]:
                                W[(d*bs + c) * n + (br_c*bs + r)][col] = c_recon[br_c][bc_c][s]
                    col += 1
    
    return {"U": U, "V": V, "W": W}
