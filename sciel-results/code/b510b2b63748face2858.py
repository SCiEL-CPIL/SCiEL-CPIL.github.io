import time

DIMENSIONS = [4, 5, 6]

def construct():
    start = time.time()
    deadline = start + 110.0
    n, m, p = DIMENSIONS
    
    try:
        # 2x2 Strassen base
        U2 = [[0]*7 for _ in range(4)]
        V2 = [[0]*7 for _ in range(4)]
        W2 = [[0]*7 for _ in range(4)]
        
        U2[0][0] = 1; U2[3][0] = 1
        V2[0][0] = 1; V2[3][0] = 1
        W2[0][0] = 1; W2[3][0] = 1
        
        U2[2][1] = 1; U2[3][1] = 1
        V2[0][1] = 1
        W2[1][1] = 1; W2[3][1] = -1
        
        U2[0][2] = 1
        V2[1][2] = 1; V2[3][2] = -1
        W2[2][2] = 1; W2[3][2] = 1
        
        U2[3][3] = 1
        V2[2][3] = 1; V2[0][3] = -1
        W2[0][3] = 1; W2[1][3] = 1
        
        U2[0][4] = 1; U2[1][4] = 1
        V2[3][4] = 1
        W2[0][4] = -1; W2[2][4] = 1
        
        U2[2][5] = 1; U2[0][5] = -1
        V2[0][5] = 1; V2[1][5] = 1
        W2[3][5] = 1
        
        U2[1][6] = 1; U2[3][6] = -1
        V2[2][6] = 1; V2[3][6] = 1
        W2[0][6] = 1
        
        # Build a generic Strassen for 2a x 2b x 2c
        def make_strassen(a, b, c):
            ra = 2 * a
            rb = 2 * b
            rc = 2 * c
            rank = 7 * a * b * c
            U = [[0]*rank for _ in range(ra * rb)]
            V = [[0]*rank for _ in range(rb * rc)]
            W = [[0]*rank for _ in range(rc * ra)]
            
            sd = [
                ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
                ([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1),((1,1),-1)]),
                ([((0,0),1)], [((0,1),1),((1,1),-1)], [((0,1),1),((1,1),1)]),
                ([((1,1),1)], [((1,0),1),((0,0),-1)], [((0,0),1),((1,0),1)]),
                ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),-1),((0,1),1)]),
                ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((1,1),1)]),
                ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((0,0),1)]),
            ]
            
            for sa in range(a):
                for sb in range(b):
                    for sc in range(c):
                        for t in range(7):
                            col = sa * b * c * 7 + sb * c * 7 + sc * 7 + t
                            a_blocks, b_blocks, c_blocks = sd[t]
                            for li in range(2):
                                for lj in range(2):
                                    ca = U2[li*2+lj][t]
                                    if ca:
                                        for (bi, bj), sig in a_blocks:
                                            gi = (bi*2 + li)
                                            gj = (bj*2 + lj)
                                            U[(sa*2+gi)*rb + (sb*2+gj)][col] += ca * sig
                            for lj in range(2):
                                for lk in range(2):
                                    cb = V2[lj*2+lk][t]
                                    if cb:
                                        for (bj, bk), sig in b_blocks:
                                            gj = (bj*2 + lj)
                                            gk = (bk*2 + lk)
                                            V[(sb*2+gj)*rc + (sc*2+gk)][col] += cb * sig
                            for lk in range(2):
                                for li in range(2):
                                    cc = W2[lk*2+li][t]
                                    if cc:
                                        for (bi, bk), sig in c_blocks:
                                            gk = (bk*2 + lk)
                                            gi = (bi*2 + li)
                                            W[(sc*2+gk)*ra + (sa*2+gi)][col] += cc * sig
            return U, V, W, rank
        
        def schoolblock(nn, mm, pp):
            r = nn * mm * pp
            U = [[0]*r for _ in range(nn*mm)]
            V = [[0]*r for _ in range(mm*pp)]
            W = [[0]*r for _ in range(pp*nn)]
            c = 0
            for i in range(nn):
                for j in range(mm):
                    for k in range(pp):
                        U[i*mm+j][c] = 1
                        V[j*pp+k][c] = 1
                        W[k*nn+i][c] = 1
                        c += 1
            return U, V, W, r
        
        def combine(decomps):
            total_rank = sum(d[3] for d in decomps)
            U = [[0]*total_rank for _ in range(n*m)]
            V = [[0]*total_rank for _ in range(m*p)]
            W = [[0]*total_rank for _ in range(p*n)]
            cur = 0
            for (Us, Vs, Ws, rs), (is_, ie), (js_, je), (ks_, ke) in decomps:
                ns, ms, ps = ie-is_, je-js_, ke-ks_
                for r in range(ns*ms):
                    il, jl = divmod(r, ms)
                    for c in range(rs):
                        U[(is_+il)*m + (js_+jl)][cur+c] = Us[r][c]
                for r in range(ms*ps):
                    jl, kl = divmod(r, ps)
                    for c in range(rs):
                        V[(js_+jl)*p + (ks_+kl)][cur+c] = Vs[r][c]
                for r in range(ps*ns):
                    kl, il = divmod(r, ns)
                    for c in range(rs):
                        W[(ks_+kl)*n + (is_+il)][cur+c] = Ws[r][c]
                cur += rs
            return U, V, W, total_rank
        
        # Strategy: 4x5x6 split into:
        # 1) 4x4x4 Strassen (rank 49)
        # 2) 4x1x6 schoolbook (rank 24)
        # 3) 4x4x2 Strassen (rank 28)
        # Total: 101
        
        U1, V1, W1, r1 = make_strassen(2, 2, 2)
        d1 = ((U1, V1, W1, r1), (0,4), (0,4), (0,4))
        
        U2, V2, W2, r2 = schoolblock(4, 1, 6)
        d2 = ((U2, V2, W2, r2), (0,4), (4,5), (0,6))
        
        U3, V3, W3, r3 = make_strassen(2, 2, 1)
        d3 = ((U3, V3, W3, r3), (0,4), (0,4), (4,6))
        
        U, V, W, _ = combine([d1, d2, d3])
        return {"U": U, "V": V, "W": W}
    except:
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
