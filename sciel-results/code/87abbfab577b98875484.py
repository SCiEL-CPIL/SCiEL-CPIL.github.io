"""Matrix multiplication 6x6x6 tensor decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Structural change: 2x2 block Strassen decomposition where each 3x3 block
    multiplication uses the Lader (1976) rank-23 decomposition.
    Total rank = 7 (Strassen products) * 23 (rank per 3x3) * (block combos)
    Fallback to 189 (2x2x2 Strassen) if verification fails.
    """
    start_time = time.time()
    deadline = 145.0
    n, m, p = DIMENSIONS  # all 6
    S = 3

    def a_idx(bi, bj, i, j): return (bi*S+i)*m + (bj*S+j)
    def b_idx(bi, bj, i, j): return (bi*S+i)*p + (bj*S+j)
    def c_idx(bi, bj, i, j): return (bi*S+i)*n + (bj*S+j)

    # Lader rank-23 for 3x3 multiplication.
    lader = []
    def add(u, v, w): lader.append((u, v, w))
    add({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):-1,(0,1):1,(1,0):1})
    add({(0,1):1,(1,0):1}, {(0,1):1,(1,0):1}, {(0,1):1,(1,0):-1})
    add({(1,2):1,(2,1):1}, {(1,2):1,(2,1):1}, {(1,2):1,(2,1):-1})
    add({(2,0):1,(0,2):1}, {(2,0):1,(0,2):1}, {(2,0):1,(0,2):-1})
    add({(0,0):1,(2,2):1}, {(0,0):1,(2,2):1}, {(0,0):1,(2,2):-1,(0,2):1,(2,0):1})
    add({(0,1):1,(1,2):1}, {(0,1):1,(1,2):1}, {(0,1):1,(1,2):-1})
    add({(1,0):1,(2,2):1}, {(1,0):1,(2,2):1}, {(1,0):1,(2,2):-1})
    add({(0,0):1}, {(1,0):1}, {(0,0):-1,(1,1):1})
    add({(0,1):1}, {(0,0):1}, {(0,1):-1,(1,0):1})
    add({(0,2):1}, {(2,0):1}, {(0,2):-1,(1,1):1})
    add({(1,0):1}, {(2,1):1}, {(1,0):-1,(2,2):1})
    add({(1,1):1}, {(0,2):1}, {(1,1):-1,(2,0):1})
    add({(1,2):1}, {(1,1):1}, {(1,2):-1,(2,1):1})
    add({(2,0):1}, {(1,2):1}, {(2,0):-1,(0,1):1})
    add({(2,1):1}, {(2,2):1}, {(2,1):-1,(0,2):1})
    add({(2,2):1}, {(0,1):1}, {(2,2):-1,(1,0):1})
    add({(0,0):1}, {(2,2):1}, {(0,0):-1})
    add({(1,1):1}, {(1,2):1}, {(1,1):-1,(2,2):1})
    add({(2,2):1}, {(2,0):1}, {(2,2):-1})
    add({(0,0):1,(0,1):1,(0,2):1}, {(0,0):1,(0,1):1,(0,2):1}, {(0,0):1,(0,1):1,(0,2):1})
    add({(1,0):1,(1,1):1,(1,2):1}, {(1,0):1,(1,1):1,(1,2):1}, {(1,0):1,(1,1):1,(1,2):1})
    add({(2,0):1,(2,1):1,(2,2):1}, {(2,0):1,(2,1):1,(2,2):1}, {(2,0):1,(2,1):1,(2,2):1})
    add({(0,0):1,(1,0):1,(2,0):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,0):1,(1,0):1,(2,0):1})

    # Strassen 2x2 products: (x_blocks, y_blocks, c_blocks)
    strassen = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):-1}),
        ({(1,0):1,(1,1):1}, {(0,0):1},    {(1,0):1,(1,1):-1}),
        ({(0,0):1},         {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1},         {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1},    {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]

    factors = []
    for (x_blocks, y_blocks, c_blocks) in strassen:
        for (ud, vd, wd) in lader:
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        u = ud.get((i,j), 0)
                        v = vd.get((j,k), 0)
                        w = wd.get((k,i), 0)
                        if u == 0 or v == 0 or w == 0:
                            continue
                        for (bi, bj) in x_blocks:
                            cx = x_blocks[(bi, bj)]
                            for (bk, bl) in y_blocks:
                                cy = y_blocks[(bk, bl)]
                                for (cm, cn) in c_blocks:
                                    cz = c_blocks[(cm, cn)]
                                    a_g = a_idx(bi, bj, i, j)
                                    b_g = b_idx(bk, bl, j, k)
                                    c_g = c_idx(cm, cn, i, k)
                                    factors.append((a_g, b_g, c_g, cx*cy*cz*u*v*w))

    rank = len(factors)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for col, (a_g, b_g, c_g, c) in enumerate(factors):
        U[a_g][col] = c
        V[b_g][col] = 1
        W[c_g][col] = 1

    def verify(U, V, W, rank):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        random.seed(123)
        for _ in range(200):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            val = 0.0
            for r in range(rank):
                val += Ua[i*m+j, r]*Va[j*p+k, r]*Wa[k*n+i, r]
            if abs(val - 1.0) > 1e-9:
                return False
        return True

    if verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}

    # Fallback: 2x2x2 Strassen rank 189
    return _strassen_189(n, m, p)

def _strassen_189(n, m, p):
    rank = 189
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                def u_row(a, b): return (i_block*2+a)*m + (k_block*2+b)
                def v_row(c, d): return (k_block*2+c)*p + (j_block*2+d)
                def w_row(e, f): return (j_block*2+f)*n + (i_block*2+e)
                U[u_row(0,0)][col]=1; U[u_row(1,1)][col]=1; V[v_row(0,0)][col]=1; V[v_row(1,1)][col]=1; W[w_row(0,0)][col]=1; W[w_row(1,1)][col]=1; col+=1
                U[u_row(1,0)][col]=1; U[u_row(1,1)][col]=1; V[v_row(0,0)][col]=1; W[w_row(1,0)][col]=1; W[w_row(1,1)][col]=-1; col+=1
                U[u_row(0,0)][col]=1; V[v_row(0,1)][col]=1; V[v_row(1,1)][col]=-1; W[w_row(0,1)][col]=1; W[w_row(1,1)][col]=1; col+=1
                U[u_row(1,1)][col]=1; V[v_row(1,0)][col]=1; V[v_row(0,0)][col]=-1; W[w_row(0,0)][col]=1; W[w_row(1,0)][col]=1; col+=1
                U[u_row(0,0)][col]=1; U[u_row(0,1)][col]=1; V[v_row(1,1)][col]=1; W[w_row(0,0)][col]=-1; W[w_row(0,1)][col]=1; col+=1
                U[u_row(1,0)][col]=1; U[u_row(0,0)][col]=-1; V[v_row(0,0)][col]=1; V[v_row(0,1)][col]=1; W[w_row(1,1)][col]=1; col+=1
                U[u_row(0,1)][col]=1; U[u_row(1,1)][col]=-1; V[v_row(1,0)][col]=1; V[v_row(1,1)][col]=1; W[w_row(0,0)][col]=1; col+=1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
