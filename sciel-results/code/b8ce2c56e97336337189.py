import time

DIMENSIONS = [2, 2, 12]

def construct():
    # Rank-42 decomposition using Strassen on six 2x2 blocks.
    # A is 2x2, B is 2x12 (six 2x2 blocks), C = A*B is 2x12.
    # U: (n*m) x rank = 4 x 42
    # V: (m*p) x rank = 24 x 42
    # W: (p*n) x rank = 24 x 42
    n, m, p = 2, 2, 12
    rank = 42
    
    # U rows indexed by (i, j) where i in 0..1, j in 0..1
    # row index = i * m + j = i * 2 + j
    # V rows indexed by (j, k) where j in 0..1, k in 0..11
    # row index = j * p + k = j * 12 + k
    # W rows indexed by (k, i) where k in 0..11, i in 0..1
    # row index = k * n + i = k * 2 + i
    
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x 42
    
    # Strassen terms for each 2x2 block:
    # For block b (columns 2b, 2b+1 of B):
    # B_block = [[x, y], [z, w]] where x=B[0,2b], y=B[0,2b+1], z=B[1,2b], w=B[1,2b+1]
    # A = [[a, b], [c, d]]
    #
    # M1 = a*(x+w)
    # M2 = (a+b)*w
    # M3 = (c+d)*x
    # M4 = d*(x-z)
    # M5 = (a+c)*w
    # M6 = (b-d)*(z+w)
    # M7 = (a+c)*(x-z)
    #
    # C[0, 2b]   = M1 + M4 - M5 + M7
    # C[0, 2b+1] = M2 - M4 - M6 + M7
    # C[1, 2b]   = M3 + M5 - M1 - M7
    # C[1, 2b+1] = M1 + M2 - M3 + M6
    
    for b in range(6):
        base = b * 7
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1 = a*(x+w): U contributions: a->U[0,0]=1
        # U row for (i=0,j=0): M1 uses a
        U[0*2+0][base+0] += 1
        # V: x+w -> V[0*12+k0]=1, V[0*12+k1]=1
        V[0*12+k0][base+0] += 1
        V[0*12+k1][base+0] += 1
        # W: M1 contributes to C[0,2b] with +1, C[1,2b+1] with +1
        W[k0*2+0][base+0] += 1   # C[0, 2b]
        W[k1*2+1][base+0] += 1   # C[1, 2b+1]
        
        # M2 = (a+b)*w: U contributions: a->U[0,0]=1, b->U[0,1]=1
        U[0*2+0][base+1] += 1
        U[0*2+1][base+1] += 1
        # V: w -> V[0*12+k1]=1
        V[0*12+k1][base+1] += 1
        # W: M2 contributes to C[0,2b+1] with +1, C[1,2b+1] with +1
        W[k1*2+0][base+1] += 1   # C[0, 2b+1]
        W[k1*2+1][base+1] += 1   # C[1, 2b+1]
        
        # M3 = (c+d)*x: U contributions: c->U[1,0]=1, d->U[1,1]=1
        U[1*2+0][base+2] += 1
        U[1*2+1][base+2] += 1
        # V: x -> V[0*12+k0]=1
        V[0*12+k0][base+2] += 1
        # W: M3 contributes to C[1,2b] with +1, C[1,2b+1] with -1
        W[k0*2+1][base+2] += 1   # C[1, 2b]
        W[k1*2+1][base+2] += -1  # C[1, 2b+1]
        
        # M4 = d*(x-z): U contributions: d->U[1,1]=1
        U[1*2+1][base+3] += 1
        # V: x-z -> V[0*12+k0]=1, V[1*12+k0]=-1
        V[0*12+k0][base+3] += 1
        V[1*12+k0][base+3] += -1
        # W: M4 contributes to C[0,2b] with +1, C[0,2b+1] with -1
        W[k0*2+0][base+3] += 1   # C[0, 2b]
        W[k1*2+0][base+3] += -1  # C[0, 2b+1]
        
        # M5 = (a+c)*w: U contributions: a->U[0,0]=1, c->U[1,0]=1
        U[0*2+0][base+4] += 1
        U[1*2+0][base+4] += 1
        # V: w -> V[0*12+k1]=1
        V[0*12+k1][base+4] += 1
        # W: M5 contributes to C[0,2b] with -1, C[1,2b] with +1
        W[k0*2+0][base+4] += -1  # C[0, 2b]
        W[k0*2+1][base+4] += 1   # C[1, 2b]
        
        # M6 = (b-d)*(z+w): U contributions: b->U[0,1]=1, d->U[1,1]=-1
        U[0*2+1][base+5] += 1
        U[1*2+1][base+5] += -1
        # V: z+w -> V[1*12+k0]=1, V[1*12+k1]=1
        V[1*12+k0][base+5] += 1
        V[1*12+k1][base+5] += 1
        # W: M6 contributes to C[0,2b+1] with -1, C[1,2b+1] with +1
        W[k1*2+0][base+5] += -1  # C[0, 2b+1]
        W[k1*2+1][base+5] += 1   # C[1, 2b+1]
        
        # M7 = (a+c)*(x-z): U contributions: a->U[0,0]=1, c->U[1,0]=1
        U[0*2+0][base+6] += 1
        U[1*2+0][base+6] += 1
        # V: x-z -> V[0*12+k0]=1, V[1*12+k0]=-1
        V[0*12+k0][base+6] += 1
        V[1*12+k0][base+6] += -1
        # W: M7 contributes to C[0,2b] with +1, C[0,2b+1] with +1, C[1,2b] with -1
        W[k0*2+0][base+6] += 1   # C[0, 2b]
        W[k1*2+0][base+6] += 1   # C[0, 2b+1]
        W[k0*2+1][base+6] += -1  # C[1, 2b]
    
    # Convert to integers where possible
    U = [[int(x) if x == int(x) else x for x in row] for row in U]
    V = [[int(x) if x == int(x) else x for x in row] for row in V]
    W = [[int(x) if x == int(x) else x for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
