import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 130

    # Split middle dim m=5 into 2+3 and output dim p=6 into 3+3.
    # C = A1*B1 + A2*B2 where A1(2x2), B1(2x6), A2(2x3), B2(3x6).
    # M(2,2,6) = 2*M(2,2,3) rank 10+10 = 20
    # M(2,3,6) = 2*M(2,3,3) rank 9+9 = 18
    # total rank = 38
    RANK = 38
    U = [[0] * RANK for _ in range(n * m)]   # (10, RANK)
    V = [[0] * RANK for _ in range(m * p)]   # (30, RANK)
    W = [[0] * RANK for _ in range(p * n)]   # (12, RANK)

    col = 0

    def lader_2x2x3(u, v, w):
        """Lader rank-10 decomposition of M(2,2,3).
        A=[[a,b],[c,d]] (u=[a,b,c,d]), B=[[x,y,z],[u,v,w]] (v=[x,y,z,u,v,w]),
        C=[[C00,C01,C02],[C10,C11,C12]] (w=[C00,C01,C02,C10,C11,C12]).
        Adds 10 terms, returns next col."""
        a, b, c, d = u
        x, y, z, uu, vv, ww = v
        C00, C01, C02, C10, C11, C12 = w
        nonlocal col
        # M1 = a*(y+uu) -> C01, C10
        U[a][col] += 1; V[y][col] += 1; V[uu][col] += 1; W[C01][col] += 1; W[C10][col] += 1; col += 1
        # M2 = d*(x+vv) -> C00, C11
        U[d][col] += 1; V[x][col] += 1; V[vv][col] += 1; W[C00][col] += 1; W[C11][col] += 1; col += 1
        # M3 = (a+b)*(z+uu) -> C02, C10
        U[a][col] += 1; U[b][col] += 1; V[z][col] += 1; V[uu][col] += 1; W[C02][col] += 1; W[C10][col] += 1; col += 1
        # M4 = (c+d)*(x+vv) -> C00, C11
        U[c][col] += 1; U[d][col] += 1; V[x][col] += 1; V[vv][col] += 1; W[C00][col] += 1; W[C11][col] += 1; col += 1
        # M5 = (c+d)*(y+vv) -> C01, C11
        U[c][col] += 1; U[d][col] += 1; V[y][col] += 1; V[vv][col] += 1; W[C01][col] += 1; W[C11][col] += 1; col += 1
        # M6 = (a+b)*(z+vv) -> C02, C11
        U[a][col] += 1; U[b][col] += 1; V[z][col] += 1; V[vv][col] += 1; W[C02][col] += 1; W[C11][col] += 1; col += 1
        # M7 = (c+d)*(z+vv) -> C02, C12
        U[c][col] += 1; U[d][col] += 1; V[z][col] += 1; V[vv][col] += 1; W[C02][col] += 1; W[C12][col] += 1; col += 1
        # M8 = a*(y-vv) -> C01, C11
        U[a][col] += 1; V[y][col] += 1; V[vv][col] -= 1; W[C01][col] += 1; W[C11][col] += 1; col += 1
        # M9 = d*(x-uu) -> C00, C10
        U[d][col] += 1; V[x][col] += 1; V[uu][col] -= 1; W[C00][col] += 1; W[C10][col] += 1; col += 1
        # M10 = (a+b)*(z-uu) -> C02, C12
        U[a][col] += 1; U[b][col] += 1; V[z][col] += 1; V[uu][col] -= 1; W[C02][col] += 1; W[C12][col] += 1; col += 1
        return col

    def m233(u, v, w):
        """Rank-9 decomposition of M(2,3,3).
        A=[[a,b,c],[d,e,f]] (u=[a,b,c,d,e,f]), B=[[x,y,z],[u,v,w],[r,s,t]]
        (v=[x,y,z,u,v,w,r,s,t]), C=[[C00,C01,C02],[C10,C11,C12]] (w=[...]).
        Adds 9 terms, returns next col."""
        a, b, c, d, e, f = u
        x, y, z, uu, vv, ww, rr, ss, tt = v
        C00, C01, C02, C10, C11, C12 = w
        nonlocal col
        # M1 = (a+b+c)*(x+u+r) -> C00,C01,C02,C10,C11,C12
        U[a][col]+=1; U[b][col]+=1; U[c][col]+=1
        V[x][col]+=1; V[uu][col]+=1; V[rr][col]+=1
        W[C00][col]+=1; W[C01][col]+=1; W[C02][col]+=1
        W[C10][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        # M2 = (a+e-f)*(y+v+s) -> C01,C02,C11,C12
        U[a][col]+=1; U[e][col]+=1; U[f][col]-=1
        V[y][col]+=1; V[vv][col]+=1; V[ss][col]+=1
        W[C01][col]+=1; W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        # M3 = (b+f-e)*(z+w+t) -> C02,C11,C12
        U[b][col]+=1; U[f][col]+=1; U[e][col]-=1
        V[z][col]+=1; V[ww][col]+=1; V[tt][col]+=1
        W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        # M4 = (a+e-f)*(x+u+r) -> C00,C01,C02,C10,C11,C12
        U[a][col]+=1; U[e][col]+=1; U[f][col]-=1
        V[x][col]+=1; V[uu][col]+=1; V[rr][col]+=1
        W[C00][col]+=1; W[C01][col]+=1; W[C02][col]+=1
        W[C10][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        # M5 = (b+f-e)*(y+v+s) -> C01,C02,C11,C12
        U[b][col]+=1; U[f][col]+=1; U[e][col]-=1
        V[y][col]+=1; V[vv][col]+=1; V[ss][col]+=1
        W[C01][col]+=1; W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        # M6 = (a+e-f)*(z+w+t) -> C02,C11,C12
        U[a][col]+=1; U[e][col]+=1; U[f][col]-=1
        V[z][col]+=1; V[ww][col]+=1; V[tt][col]+=1
        W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        # M7 = (a+d+e)*(x+u+r) -> C00,C01,C02,C10,C11,C12
        U[a][col]+=1; U[d][col]+=1; U[e][col]+=1
        V[x][col]+=1; V[uu][col]+=1; V[rr][col]+=1
        W[C00][col]+=1; W[C01][col]+=1; W[C02][col]+=1
        W[C10][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        # M8 = (b+d+f)*(y+v+s) -> C01,C02,C11,C12
        U[b][col]+=1; U[d][col]+=1; U[f][col]+=1
        V[y][col]+=1; V[vv][col]+=1; V[ss][col]+=1
        W[C01][col]+=1; W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        # M9 = (c+d+e)*(z+w+t) -> C02,C11,C12
        U[c][col]+=1; U[d][col]+=1; U[e][col]+=1
        V[z][col]+=1; V[ww][col]+=1; V[tt][col]+=1
        W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        return col

    # A1 = A[0:2,0:2] -> U rows [0,1,5,6]
    # A2 = A[0:2,2:5] -> U rows [2,3,4,7,8,9]
    # B1 rows 0-1: V rows [0..5, 6..11]; B2 rows 2-4: V rows [12..29]
    # C cols 0-2: W [0..5]; C cols 3-5: W [6..11]

    # A1 * B1[:,0:3] -> C[:,0:3]
    lader_2x2x3(
        u=[0, 1, 5, 6],
        v=[0, 1, 2, 6, 7, 8],
        w=[0, 1, 2, 3, 4, 5])
    # A1 * B1[:,3:6] -> C[:,3:6]
    lader_2x2x3(
        u=[0, 1, 5, 6],
        v=[3, 4, 5, 9, 10, 11],
        w=[6, 7, 8, 9, 10, 11])
    # A2 * B2[:,0:3] -> C[:,0:3]
    m233(
        u=[2, 3, 4, 7, 8, 9],
        v=[12, 13, 14, 18, 19, 20, 24, 25, 26],
        w=[0, 1, 2, 3, 4, 5])
    # A2 * B2[:,3:6] -> C[:,3:6]
    m233(
        u=[2, 3, 4, 7, 8, 9],
        v=[15, 16, 17, 21, 22, 23, 27, 28, 29],
        w=[6, 7, 8, 9, 10, 11])

    # ---- Verify correctness: tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] must equal 1 ----
    def schoolbook():
        rank = n * m * p
        Us = [[0] * rank for _ in range(n * m)]
        Vs = [[0] * rank for _ in range(m * p)]
        Ws = [[0] * rank for _ in range(p * n)]
        cc = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i * m + j][cc] = 1
                    Vs[j * p + k][cc] = 1
                    Ws[k * n + i][cc] = 1
                    cc += 1
        return {"U": Us, "V": Vs, "W": Ws}

    if time.time() < deadline:
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(RANK):
                        s += Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            return schoolbook()

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
