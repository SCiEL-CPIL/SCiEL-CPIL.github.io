import numpy as np

def gen(n, m, p):
    n, m, p = int(n), int(m), int(p)
    U = {}
    V = {}
    W = {}
    def add(c, ui, vj, wk, s):
        if s == 0:
            return
        U[(ui, c)] = s
        V[(vj, c)] = 1
        W[(wk, c)] = 1
    # Strassen core on the 4x4 x 4x4 part (first 4 cols of B)
    M = [[(0,0),(1,1)],[(0,1),(1,0)],[(0,0),(1,0)],[(0,1),(1,1)]]
    S = [[(0,0),(1,1)],[(0,0),(1,0)],[(0,0),(1,1)],[(1,0),(1,1)]]
    for (i,j),(a,b) in M:
        for (k,l),(c,d) in S:
            add((i,j,k,l), (i,a), (j,c), (l,b), d)
    # Schoolbook for the extra 2 columns (cols 4,5 of B)
    for j in range(m):
        for k in range(4, p):
            for i in range(n):
                add((i,j,k), (i,j), (j,k), (k,i), 1)
    cols = sorted(U.keys())
    U = [[0]*len(cols) for _ in range(n*m)]
    V = [[0]*len(cols) for _ in range(m*p)]
    W = [[0]*len(cols) for _ in range(p*n)]
    for c, key in enumerate(cols):
        (ui, vj, wk, s) = (key[0], key[1], key[2], U[(ui, key)] if (ui,key) in U else 0)
    return U, V, W, cols
