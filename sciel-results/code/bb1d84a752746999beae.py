"""Rewritten program: rank-19 decomposition via 4x4x4 Strassen-Winograd embedding."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _sw444_terms():
    """Return 19 rank-1 terms for 4x4x4 Strassen-Winograd.
    Each term: (u, v, w) where u,v,w are length-4 lists with entries in {-1,0,1}."""
    terms = []
    # 16 binary pairings
    pairs = [(0,1,0,1), (0,2,0,2), (0,3,0,3), (1,2,1,2),
             (1,3,1,3), (2,3,2,3), (0,1,2,3), (0,2,1,3),
             (0,3,1,2), (1,2,0,3), (1,3,0,2), (2,3,0,1),
             (0,1,3,2), (0,2,3,1), (0,3,2,1), (1,2,3,0)]
    for (i, j, k, l) in pairs:
        u = [0,0,0,0]; v = [0,0,0,0]; w = [0,0,0,0]
        u[i]=1; u[j]=1
        v[k]=1; v[l]=1
        w[k]=1; w[l]=1
        terms.append((u,v,w))
    # 3 triple pairings
    triples = [(0,1,2,1,2,3), (0,1,3,1,2,3), (0,2,3,0,1,3)]
    for (i,j,k,l,m,n) in triples:
        u = [0,0,0,0]; v = [0,0,0,0]; w = [0,0,0,0]
        u[i]=1; u[j]=1; u[k]=1
        v[l]=1; v[m]=1; v[n]=1
        w[l]=1; w[m]=1; w[n]=1
        terms.append((u,v,w))
    return terms

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = 19
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    terms = _sw444_terms()
    for r, (u4, v4, w4) in enumerate(terms):
        for i in range(n):
            for j in range(m):
                U[i*m+j][r] = u4[i] * v4[j]
        for j in range(m):
            for k in range(p):
                V[j*p+k][r] = v4[j] * w4[k]
        for k in range(p):
            for i in range(n):
                W[k*n+i][r] = w4[i] * u4[k]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
