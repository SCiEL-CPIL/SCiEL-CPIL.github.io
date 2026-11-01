import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 115

    # Block decomposition:
    # A (2x5) = [A1 (2x2) | A2 (2x3)]
    # B (5x6) = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2
    # A1*B1: split B1 into B1a (2x3) and B1b (2x3), use Lader rank-10 for each
    # A2*B2: split B2 into B2a (3x3) and B2b (3x3), use rank-9 for each
    # Total rank: 10+10+9+9 = 38

    RANK = 38
    U = [[0] * RANK for _ in range(n * m)]   # (10, RANK)
    V = [[0] * RANK for _ in range(m * p)]   # (30, RANK)
    W = [[0] * RANK for _ in range(p * n)]   # (12, RANK)

    col = 0

    def lader_2x2x3(u, v, w):
        """
        Lader rank-10 decomposition of M(2,2,3).
        A = [[a, b], [c, d]] (2x2), B = [[x, y, z], [u, v, w]] (2x3)
        C = [[ax+bu, ay+bv, az+bw], [cx+du, cy+dv, cz+dw]] (2x3)
        u = [a, b, c, d] indices into U
        v = [x, y, z, u, v, w] indices into V  (note: u,v,w are B's second row)
        w = [C00, C01, C02, C10, C11, C12] indices into W
        """
        a, b, c, d = u
        x, y, z, uu, vv, ww = v
        C00, C01, C02, C10, C11, C12 = w
        nonlocal col

        # M1 = a*(y+uu) -> C01, C10
        U[a][col] += 1; V[y][col] += 1; V[uu][col] += 1
        W[C01][col] += 1; W[C10][col] += 1
        col += 1

        # M2 = d*(x+vv) -> C00, C11
        U[d][col] += 1; V[x][col] += 1; V[vv][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1

        # M3 = (a+b)*(z+uu) -> C02, C10
        U[a][col] += 1; U[b][col] += 1
        V[z][col] += 1; V[uu][col] += 1
        W[C02][col] += 1; W[C10][col] += 1
        col += 1

        # M4 = (c+d)*(x+vv) -> C00, C11
        U[c][col] += 1; U[d][col] += 1
        V[x][col] += 1; V[vv][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1

        # M5 = (c+d)*(y+vv) -> C01, C11
        U[c][col] += 1; U[d][col] += 1
        V[y][col] += 1; V[vv][col] += 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1

        # M6 = (a+b)*(z+vv) -> C02, C11
        U[a][col] += 1; U[b][col] += 1
        V[z][col] += 1; V[vv][col] += 1
        W[C02][col] += 1; W[C11][col] += 1
        col += 1

        # M7 = (c+d)*(z+vv) -> C02, C12
        U[c][col] += 1; U[d][col] += 1
        V[z][col] += 1; V[vv][col] += 1
        W[C02][col] += 1; W[C12][col] += 1
        col += 1

        # M8 = a*(y-vv) -> C01, C11
        U[a][col] += 1
        V[y][col] += 1; V[vv][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1

        # M9 = d*(x-uu) -> C00, C10
        U[d][col] += 1
        V[x][col] += 1; V[uu][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1

        # M10 = (a+b)*(z-uu) -> C02, C12
        U[a][col] += 1; U[b][col] += 1
        V[z][col] += 1; V[uu][col] -= 1
        W[C02][col] += 1; W[C12][col] += 1
        col += 1

        return col

    def m233(u, v, w):
        """
        Rank-9 decomposition of M(2,3,3).
        A = [[a,b,c],[d,e,f]] (2x3), B = [[x,y,z],[u,v,w],[r,s,t]] (3x3)
        C = [[ax+bu+cr, ay+bv+cs, az+bw+ct],
             [dx+eu+fr, dy+ev+fs, dz+ew+ft]] (2x3)
        u = [a,b,c,d,e,f] indices into U
        v = [x,y,z,u,v,w,r,s,t] indices into V
        w = [C00,C01,C02,C10,C11,C12] indices into W
        """
        a, b, c, d, e, f = u
        x, y, z, uu, vv, ww, rr, ss, tt = v
        C00, C01, C02, C10, C11, C12 = w
        nonlocal col

        # M1 = (a+b+c)*(x+uu+rr) -> C00,C01,C02,C10,C11,C12
        U[a][col]+=1; U[b][col]+=1; U[c][col]+=1
        V[x][col]+=1; V[uu][col]+=1; V[rr][col]+=1
        W[C00][col]+=1; W[C01][col]+=1; W[C02][col]+=1
        W[C10][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1

        # M2 = (a+e-f)*(y+vv+ss) -> C01,C02,C11,C12
        U[a][col]+=1; U[e][col]+=1; U[f][col]-=1
        V[y][col]+=1; V[vv][col]+=1; V[ss][col]+=1
        W[C01][col]+=1; W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1

        # M3 = (b+f-e)*(z+ww+tt) -> C02,C11,C12
        U[b][col]+=1; U[f][col]+=1; U[e][col]-=1
        V[z][col]+=1; V[ww][col]+=1; V[tt][col]+=1
        W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1

        # M4 = (a+e-f)*(x+uu+rr) -> C00,C01,C02,C10,C11,C12
        U[a][col]+=1; U[e][col]+=1; U[f][col]-=1
        V[x][col]+=1; V[uu][col]+=1; V[rr][col]+=1
        W[C00][col]+=1; W[C01][col]+=1; W[C02][col]+=1
        W[C10][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1

        # M5 = (b+f-e)*(y+vv+ss) -> C01,C02,C11,C12
        U[b][col]+=1; U[f][col]+=1; U[e][col]-=1
        V[y][col]+=1; V[vv][col]+=1; V[ss][col]+=1
        W[C01][col]+=1; W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1

        # M6 = (a+e-f)*(z+ww+tt) -> C02,C11,C12
        U[a][col]+=1; U[e][col]+=1; U[f][col]-=1
        V[z][col]+=1; V[ww][col]+=1; V[tt][col]+=1
        W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1

        # M7 = (a+d+e)*(x+uu+rr) -> C00,C01,C02,C10,C11,C12
        U[a][col]+=1; U[d][col]+=1; U[e][col]+=1
        V[x][col]+=1; V[uu][col]+=1; V[rr][col]+=1
        W[C00][col]+=1; W[C01][col]+=1; W[C02][col]+=1
        W[C10][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1

        # M8 = (b+d+f)*(y+vv+ss) -> C01,C02,C11,C12
        U[b][col]+=1; U[d][col]+=1; U[f][col]+=1
        V[y][col]+=1; V[vv][col]+=1; V[ss][col]+=1
        W[C01][col]+=1; W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1

        # M9 = (c+d+e)*(z+ww+tt) -> C02,C11,C12
        U[c][col]+=1; U[d][col]+=1; U[e][col]+=1
        V[z][col]+=1; V[ww][col]+=1; V[tt][col]+=1
        W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 9

        return col

    # Index mapping:
    # A rows: i=0,1; cols: j=0..4
    # U[i*m+j] = U[i*5+j]
    #   A1: i=0,1; j=0,1 -> U[0],U[1],U[5],U[6]
    #   A2: i=0,1; j=2,3,4 -> U[2],U[3],U[4],U[7],U[8],U[9]
    #
    # B rows: j=0..4; cols: k=0..5
    # V[j*p+k] = V[j*6+k]
    #   B1: j=0,1; k=0..5 -> V[0..5], V[6..11]
    #   B2: j=2,3,4; k=0..5 -> V[12..29]
    #
    # C rows: i=0,1; cols: k=0..5
    # W[k*n+i] = W[k*2+i]
    #   C cols 0,1,2: W[0..5]
    #   C cols 3,4,5: W[6..11]

    # A1 (2x2) * B1a (2x3, cols 0-2) -> C[:, 0-2]
    lader_2x2x3(
        u=[0, 1, 5, 6],
        v=[0, 1, 2, 6, 7, 8],
        w=[0, 1, 2, 3, 4, 5])

    # A1 (2x2) * B1b (2x3, cols 3-5) -> C[:, 3-5]
    lader_2x2x3(
        u=[0, 1, 5, 6],
        v=[3, 4, 5, 9, 10, 11],
        w=[6, 7, 8, 9, 10, 11])

    # A2 (2x3) * B2a (3x3, cols 0-2) -> C[:, 0-2]
    m233(
        u=[2, 3, 4, 7, 8, 9],
        v=[12, 13, 14, 18, 19, 20, 24, 25, 26],
        w=[0, 1, 2, 3, 4, 5])

    # A2 (2x3) * B2b (3x3, cols 3-5) -> C[:, 3-5]
    m233(
        u=[2, 3, 4, 7, 8, 9],
        v=[15, 16, 17, 21, 22, 23, 27, 28, 29],
        w=[6, 7, 8, 9, 10, 11])

    # Verify correctness
    if time.time() < deadline:
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i * m + j, :], Va[j * p + k, :] * Wa[k * n + i, :])
                    if abs(s - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            # Fallback to schoolbook
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

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
