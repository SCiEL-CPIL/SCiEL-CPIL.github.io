def construct():
    n, m, p = 3, 4, 6
    rank = 28
    
    U = [[0.0 for _ in range(rank)] for _ in range(n*m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m*p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p*n)]
    
    r = 0
    
    # Define 2x2x3 decomposition vectors
    vecs_223 = [
        ([1,0], [1,0], [1,0,0]),
        ([1,0], [0,1], [1,0,0]),
        ([0,1], [1,0], [1,0,0]),
        ([0,1], [0,1], [1,0,0]),
        ([1,1], [1,1], [0,1,0]),
        ([1,1], [1,1], [0,0,1])
    ]
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1,2}
    for u, v, w in vecs_223:
        for ii in range(2):
            for jj in range(2):
                U[ii*4 + jj][r] = u[ii] * v[jj]  # Wait, no. u is indexed by i', v by j'
        # Actually, U_{(i,j),r} = u_i * ... no, that's not right.
