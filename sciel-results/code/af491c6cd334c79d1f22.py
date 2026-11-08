import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    def zero_u(): return [0] * (n * m)
    def zero_v(): return [0] * (m * p)
    def zero_w(): return [0] * (p * n)
    
    # Strassen 2x2 formula templates (local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1))
    S2 = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    # Part 1: Strassen 4x4x4 for cols 0-3 (49 terms)
    for ou, ov, ow in S2:
        for iu, iv, iw in S2:
            u = zero_u(); v = zero_v(); w = zero_w()
            for (lu, cu) in iu:
                for (bu, cb) in ou:
                    br, bc = bu // 2, bu % 2
                    lr, lc = lu // 2, lu % 2
                    r, c = br*2+lr, bc*2+lc
                    u[r*m+c] += cb * cu
            for (lv, cv) in iv:
                for (bv, cb) in ov:
                    br, bc = bv // 2, bv % 2
                    lr, lc = lv // 2, lv % 2
                    r, c = br*2+lr, bc*2+lc
                    v[r*p+c] += cb * cv
            for (lw, cw) in iw:
                for (bw, cb) in ow:
                    br, bc = bw // 2, bw % 2
                    lr, lc = lw // 2, lw % 2
                    r, c = br*2+lr, bc*2+lc
                    w[c*n+r] += cb * cw
            add_term(u, v, w)
    
    # Part 2: Strassen 4x4x2 for cols 4-5 (28 terms)
    for i_b in range(2):
        for j_b in range(2):
            i0, j0, k0 = i_b*2, j_b*2, 4
            for (iu, iv, iw) in S2:
                u = zero_u(); v = zero_v(); w = zero_w()
                for (lu, cu) in iu:
                    lr, lc = lu // 2, lu % 2
                    u[(i0+lr)*m + (j0+lc)] = cu
                for (lv, cv) in iv:
                    lr, lc = lv // 2, lv % 2
                    v[(j0+lr)*p + (k0+lc)] = cv
                for (lw, cw) in iw:
                    lr, lc = lw // 2, lw % 2
                    w[(k0+lc)*n + (i0+lr)] = cw
                add_term(u, v, w)
    
    # Part 3: Schoolbook 4x4x1 for col 6 (16 terms)
    for i in range(n):
        for j in range(m):
            u = zero_u(); v = zero_v(); w = zero_w()
            u[i*m+j] = 1
            v[j*p+6] = 1
            w[6*n+i] = 1
            add_term(u, v, w)
    
    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}
    
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n*m):
            if u[r]: U[r][c] = u[r]
        for r in range(m*p):
            if v[r]: V[r][c] = v[r]
        for r in range(p*n):
            if w[r]: W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
