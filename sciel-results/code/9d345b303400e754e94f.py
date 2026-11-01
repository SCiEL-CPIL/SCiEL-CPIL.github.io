"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 3
    
    terms = []
    
    def zeros3():
        return [[0,0,0] for _ in range(3)]
    
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    # Strassen 2x2 core (embedded in 3x3)
    # u1, v1, w1
    u = zeros3(); v = zeros3(); w = zeros3()
    u[0][0]=1; u[1][1]=1
    v[0][0]=1; v[1][1]=1
    w[0][0]=1; w[1][1]=1
    add_term(u,v,w)
    
    # u2, v2, w2
    u = zeros3(); v = zeros3(); w = zeros3()
    u[1][0]=1; u[1][1]=1
    v[0][0]=1
    w[0][1]=1
    add_term(u,v,w)
    
    # u3, v3, w3
    u = zeros3(); v = zeros3(); w = zeros3()
    u[0][0]=1
    v[0][1]=1; v[1][1]=-1
    w[1][0]=1
    add_term(u,v,w)
    
    # u4, v4, w4
    u = zeros3(); v = zeros3(); w = zeros3()
    u[1][1]=1
    v[0][0]=-1; v[1][0]=1
    w[0][1]=1
    add_term(u,v,w)
    
    # u5, v5, w5
    u = zeros3(); v = zeros3(); w = zeros3()
    u[0][0]=1; u[0][1]=1
    v[1][1]=1
    w[0][1]=1
    add_term(u,v,w)
    
    # u6, v6, w6
    u = zeros3(); v = zeros3(); w = zeros3()
    u[0][0]=-1; u[1][0]=1
    v[0][0]=1; v[0][1]=1
    w[1][1]=1
    add_term(u,v,w)
    
    # u7, v7, w7
    u = zeros3(); v = zeros3(); w = zeros3()
    u[0][1]=-1; u[1][1]=1
    v[1][0]=1; v[1][1]=1
    w[0][0]=-1
    add_term(u,v,w)
    
    # Cross terms: all (i,j,k) with at least one index = 2
    cross = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i==2 or j==2 or k==2:
                    cross.append((i,j,k))
    
    for (i,j,k) in cross:
        u = zeros3(); v = zeros3(); w = zeros3()
        u[i][j]=1
        v[j][k]=1
        w[k][i]=1
        add_term(u,v,w)
    
    rank = len(terms)
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    for r, (u, v, w) in enumerate(terms):
        for i in range(3):
            for j in range(3):
                if u[i][j] != 0:
                    U[i*3+j][r] = u[i][j]
        for j in range(3):
            for k in range(3):
                if v[j][k] != 0:
                    V[j*3+k][r] = v[j][k]
        for k in range(3):
            for i in range(3):
                if w[k][i] != 0:
                    W[k*3+i][r] = w[k][i]
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                val = 0
                for r in range(rank):
                    val += U_arr[i*3+j, r] * V_arr[j*3+k, r] * W_arr[k*3+i, r]
                if abs(val - 1) > 1e-9:
                    raise ValueError(f"Invalid at ({i},{j},{k}): {val}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
