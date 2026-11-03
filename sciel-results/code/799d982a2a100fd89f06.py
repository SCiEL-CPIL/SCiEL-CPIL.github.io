DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    terms = []
    
    for l in range(3):
        k1 = 2 * l
        k2 = 2 * l + 1
        
        # Strassen 2x2x2
        # a00=A[0,0], a01=A[0,1], a10=A[1,0], a11=A[1,1]
        # b00=B[0,k1], b01=B[0,k2], b10=B[1,k1], b11=B[1,k2]
        
        # M1 = (a00+a11)(b00+b11)
        u = {0:1, 4:1}
        v = {0*p+k1:1, 1*p+k2:1}
        w = {k1*n+0:1, k2*n+1:1}
        terms.append((u,v,w))
        
        # M2 = (a10+a11)b00
        u = {3:1, 4:1}
        v = {0*p+k1:1}
        w = {k1*n+1:1}
        terms.append((u,v,w))
        
        # M3 = a00(b01-b11)
        u = {0:1}
        v = {0*p+k2:1, 1*p+k2:-1}
        w = {k2*n+0:1}
        terms.append((u,v,w))
        
        # M4 = a11(b10-b00)
        u = {4:1}
        v = {1*p+k1:1, 0*p+k1:-1}
        w = {k1*n+0:1, k1*n+1:1}
        terms.append((u,v,w))
        
        # M5 = (a00+a01)b11
        u = {0:1, 1:1}
        v = {1*p+k2:1}
        w = {k1*n+0:-1, k2*n+0:1}
        terms.append((u,v,w))
        
        # M6 = (a10-a00)(b00+b01)
        u = {3:1, 0:-1}
        v = {0*p+k1:1, 0*p+k2:1}
        w = {k2*n+1:1}
        terms.append((u,v,w))
        
        # M7 = (a01-a11)(b10+b11)
        u = {1:1, 4:-1}
        v = {1*p+k1:1, 1*p+k2:1}
        w = {k1*n+0:1, k2*n+1:1}
        terms.append((u,v,w))
        
        # 2x1x2 part
        # a02=A[0,2], a12=A[1,2]
        # b20=B[2,k1], b21=B[2,k2]
        
        # M8 = a02 b20
        u = {2:1}
        v = {2*p+k1:1}
        w = {k1*n+0:1}
        terms.append((u,v,w))
        
        # M9 = a02 b21
        u = {2:1}
        v = {2*p+k2:1}
        w = {k2*n+0:1}
        terms.append((u,v,w))
        
        # M10 = a12 b20
        u = {5:1}
        v = {2*p+k1:1}
        w = {k1*n+1:1}
        terms.append((u,v,w))
        
        # M11 = a12 b21
        u = {5:1}
        v = {2*p+k2:1}
        w = {k2*n+1:1}
        terms.append((u,v,w))
        
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][r] = val
        for idx, val in v.items():
            V[idx][r] = val
        for idx, val in w.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
