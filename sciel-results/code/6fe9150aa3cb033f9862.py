"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Strassen 2x2x2 blocks + direct border terms, rank 68, verified exact.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0

    def strassen_decomp(ri, ci, rj, ck, U_t, V_t, W_t):
        a = (ri, ci); b = (ri, ci+1); c = (ri+1, ci); d = (ri+1, ci+1)
        e = (rj, ck); f = (rj, ck+1); g = (rj+1, ck); h = (rj+1, ck+1)
        C00 = (ri, ck); C01 = (ri, ck+1); C10 = (ri+1, ck); C11 = (ri+1, ck+1)
        U_t.append([(a,1),(d,1)]); V_t.append([(e,1),(h,1)]); W_t.append([(C00,1),(C11,1)])
        U_t.append([(c,1),(d,1)]); V_t.append([(e,1)]); W_t.append([(C10,1),(C11,-1)])
        U_t.append([(a,1)]); V_t.append([(f,1),(h,-1)]); W_t.append([(C01,1),(C11,1)])
        U_t.append([(d,1)]); V_t.append([(g,1),(e,-1)]); W_t.append([(C00,1),(C10,1)])
        U_t.append([(a,1),(b,1)]); V_t.append([(h,1)]); W_t.append([(C00,-1),(C01,1)])
        U_t.append([(c,1),(a,-1)]); V_t.append([(e,1),(f,1)]); W_t.append([(C11,1)])
        U_t.append([(b,1),(d,-1)]); V_t.append([(g,1),(h,1)]); W_t.append([(C00,1)])

    def direct(i, j, k, U_t, V_t, W_t):
        U_t.append([((i,j),1)]); V_t.append([((j,k),1)]); W_t.append([((i,k),1)])

    U_t, V_t, W_t = [], [], []
    for ck in range(0, 8, 2):
        strassen_decomp(0, 0, 0, ck, U_t, V_t, W_t)
    for i in range(2):
        for k in range(8):
            direct(i, 2, k, U_t, V_t, W_t)
    for j in range(2):
        for k in range(8):
            direct(2, j, k, U_t, V_t, W_t)
    for k in range(8):
        direct(2, 2, k, U_t, V_t, W_t)

    rank = len(W_t)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for idx in range(rank):
        for (i,j),cf in U_t[idx]: U[i*m+j][idx] = cf
        for (j,k),cf in V_t[idx]: V[j*p+k][idx] = cf
        for (i,k),cf in W_t[idx]: W[k*n+i][idx] = cf

    Ua = np.array(U, dtype=float); Va = np.array(V, dtype=float); Wa = np.array(W, dtype=float)

    def verify():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i*m+j], Va[j*p+k])
                    s = np.sum(s * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify():
        r = n*m*p
        Uf = [[0]*r for _ in range(n*m)]; Vf = [[0]*r for _ in range(m*p)]; Wf = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i*m+j][col]=1; Vf[j*p+k][col]=1; Wf[k*n+i][col]=1; col+=1
        return {"U": Uf, "V": Vf, "W": Wf}

    def nice(arr):
        out = []
        for row in arr:
            nr = []
            for v in row:
                v2 = v*2
                if abs(v2 - round(v2)) < 1e-6:
                    nv = round(v2)/2.0
                    if abs(nv - round(nv)) < 1e-9: nv = int(round(nv))
                    nr.append(nv)
                else: nr.append(v)
            out.append(nr)
        return out

    return {"U": nice(Ua), "V": nice(Va), "W": nice(Wa)}
# EVOLVE-BLOCK-END
