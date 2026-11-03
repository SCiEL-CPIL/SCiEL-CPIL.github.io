DIMENSIONS = [2, 3, 3]

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, trials=40):
    import random
    rank = len(U[0])
    for _ in range(trials):
        A = [[random.randint(-3,3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3,3) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j]*B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    for r in range(rank):
                        s += U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r]
                if s != C[i][k]:
                    return False
    return True

def _rank17(n, m, p):
    rank = 17
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    def a(i,j): return i*m+j
    def b(j,k): return j*p+k
    def c(k,i): return k*n+i
    A11,A12,A13 = a(0,0),a(0,1),a(0,2)
    A21,A22,A23 = a(1,0),a(1,1),a(1,2)
    B11,B12,B13 = b(0,0),b(0,1),b(0,2)
    B21,B22,B23 = b(1,0),b(1,1),b(1,2)
    B31,B32,B33 = b(2,0),b(2,1),b(2,2)
    C00,C01,C02 = c(0,0),c(1,0),c(2,0)
    C10,C11,C12 = c(0,1),c(1,1),c(2,1)
    r = 0
    def add(ae,be,ce):
        nonlocal r
        for idx,cf in ae: U[idx][r]+=cf
        for idx,cf in be: V[idx][r]+=cf
        for idx,cf in ce: W[idx][r]+=cf
        r+=1
    # Strassen M's (CLRS)
    add([(A11,1),(A22,1)],[(B11,1),(B22,1)],[(C00,1),(C11,1)])          # M1 -> Z11,Z22
    add([(A21,1),(A22,1)],[(B11,1)],[(C01,1),(C11,-1)])                   # M2 -> Z12, -Z22
    add([(A11,1),(A21,-1)],[(B21,1),(B22,1)],[(C10,1),(C11,1)])           # M3 -> Z21,Z22
    add([(A12,1),(A22,1)],[(B11,1),(B12,-1)],[(C00,1),(C11,1)])           # M4 -> Z11,Z22
    add([(A11,1)],[(B12,1),(B22,-1)],[(C00,-1),(C01,1)])                  # M5 -> -Z11,Z12
    add([(A22,1)],[(B21,1),(B11,-1)],[(C10,1)])                           # M6 -> Z21
    add([(A11,1),(A12,1)],[(B21,1),(B11,-1)],[(C00,1)])                   # M7 -> Z11
    # Aa * Ba2
    add([(A11,1)],[(B13,1)],[(C02,1)])
    add([(A12,1)],[(B23,1)],[(C02,1)])
    add([(A21,1)],[(B13,1)],[(C12,1)])
    add([(A22,1)],[(B23,1)],[(C12,1)])
    # Ab * Bb
    add([(A13,1)],[(B31,1)],[(C00,1)])
    add([(A13,1)],[(B32,1)],[(C01,1)])
    add([(A13,1)],[(B33,1)],[(C02,1)])
    add([(A23,1)],[(B31,1)],[(C10,1)])
    add([(A23,1)],[(B32,1)],[(C11,1)])
    add([(A23,1)],[(B33,1)],[(C12,1)])
    assert r == rank
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    cand = _rank17(n, m, p)
    if _verify(cand["U"], cand["V"], cand["W"], n, m, p):
        return cand
    return _schoolbook(n, m, p)
