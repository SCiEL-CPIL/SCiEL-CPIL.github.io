import time
from collections import defaultdict

DIMENSIONS = [6, 6, 6]

def construct():
    t0 = time.time()
    n = m = p = 6
    bs = 3
    # 2x2 Strassen at top level (3x3 blocks), schoolbook 3x3 inside.
    # Each entry: (cA, bi, bj) = coeff, A-block-row, A-block-col; similarly B.
    strassen = [
        ([(1,0,0),(1,1,1)], [(1,0,0),(1,1,1)]),   # M1=(A11+A22)(B11+B22)
        ([(1,1,0),(1,1,1)], [(1,0,0)]),            # M2=(A21+A22)B11
        ([(1,0,0)], [(1,0,1),(-1,1,1)]),           # M3=A11(B12-B22)
        ([(1,1,1)], [(1,1,0),(-1,0,0)]),           # M4=A22(B21-B11)
        ([(1,0,0),(1,0,1)], [(1,1,1)]),            # M5=(A11+A12)B22
        ([(1,1,0),(-1,0,0)], [(1,0,0),(1,0,1)]),   # M6=(A21-A11)(B11+B12)
        ([(1,0,1),(-1,1,1)], [(1,1,0),(1,1,1)]),   # M7=(A12-A22)(B21+B22)
    ]
    combo = defaultdict(int)
    for cA_list, cB_list in strassen:
        for cA, bi, bj in cA_list:
            for cB, bk, bl in cB_list:
                if bj != bk:
                    continue  # A column block must equal B row block
                for li in range(bs):
                    for lj in range(bs):
                        for ll in range(bs):
                            u = (bi*bs+li)*6 + (bj*bs+lj)
                            v = (bk*bs+lj)*6 + (bl*bs+ll)
                            w = (bl*bs+ll)*6 + (bi*bs+li)
                            combo[(u, v, w)] += cA * cB
    final = [(u, v, w, c) for (u, v, w), c in combo.items() if c != 0]
    rank = len(final)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for r, (u, v, w, c) in enumerate(final):
        U[u][r] = c
        V[v][r] = c
        W[w][r] = c
    return {"U": U, "V": V, "W": W}
