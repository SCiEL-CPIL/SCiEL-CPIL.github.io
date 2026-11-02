def construct():
    n, m, p = 2, 4, 6
    rank = 30
    
    U = [[0]*rank for _ in range(n*m)]  # (8, 30)
    V = [[0]*rank for _ in range(m*p)]  # (24, 30)
    W = [[0]*rank for _ in range(p*n)]  # (12, 30)
    
    # (2,2,6) decomposition: 15 terms
    # U_sub: 2x2 (i, j_local)
    # V_sub: 2x6 (j_local, k)
    # W_sub: 6x2 (k, i)
    
    strassen = [
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]),
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,0],[1,0]]),
        ([[0,0],[0,1]], [[0,-1],[1,0]], [[0,0],[1,0]]),
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[0,0]]),
        ([[0,-1],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        ([[0,1],[-1,0]], [[0,0],[1,1]], [[1,0],[0,0]]),
    ]
    
    tails = []
    for k0 in range(2, 6):
        Vs = [[1 if k==k0 else 0 for k in range(6)] for _ in range(2)]
        Ws = [[1 if k==k0 else 0, 0] if False else ([1,0] if k==k0 else [0,0]) for k in range(6)]
        tails.append(([[1,1],[0,0]], Vs, Ws))
        Ws1 = [[0, 1 if k==k0 else 0] if False else ([0,1] if k==k0 else [0,0]) for k in range(6)]
        tails.append(([[0,0],[1,1]], Vs, Ws1))
    
    terms = strassen + tails
    assert len(terms) == 15
    
    for r, (Us, Vs, Ws) in enumerate(terms):
        # Left half: j_local=0,1 → j=0,1
        for i in range(2):
            for jl in range(2):
                U[i*4 + jl][r] = Us[i][jl]
        for jl in range(2):
            for k in range(6):
                V[jl*6 + k][r] = Vs[jl][k]
        for k in range(6):
            for i in range(2):
                W[k*2 + i][r] = Ws[k][i]
        
        # Right half: j_local=0,1 → j=2,3
        r2 = r + 15
        for i in range(2):
            for jl in range(2):
                U[i*4 + (jl+2)][r2] = Us[i][jl]
        for jl in range(2):
            for k in range(6):
                V[(jl+2)*6 + k][r2] = Vs[jl][k]
        for k in range(6):
            for i in range(2):
                W[k*2 + i][r2] = Ws[k][i]
    
    return {"U": U, "V": V, "W": W}
