DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    strassen = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        ([0,0,0,1], [0,-1,1,0], [1,0,0,1]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([0,-1,1,0], [1,0,1,0], [0,0,0,1]),
        ([0,1,0,-1], [0,1,0,1], [1,0,0,0]),
    ]
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i1 in range(2):
        for k1 in range(3):
            for j1 in range(2):
                for (u_vec, v_vec, w_vec) in strassen:
                    for i2 in range(2):
                        for j2 in range(2):
                            u_idx = 8*i1 + 4*i2 + 2*j1 + j2
                            U[u_idx][col] += u_vec[i2*2 + j2]
                    for j2 in range(2):
                        for k2 in range(2):
                            v_idx = 12*j1 + 6*j2 + 2*k1 + k2
                            V[v_idx][col] += v_vec[j2*2 + k2]
                    for i2 in range(2):
                        for k2 in range(2):
                            w_idx = 8*k1 + 4*k2 + 2*i1 + i2
                            W[w_idx][col] += w_vec[i2*2 + k2]
                    col += 1
    
    return {"U": U, "V": V, "W": W}
