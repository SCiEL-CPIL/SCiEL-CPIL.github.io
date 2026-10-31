import time

DIMENSIONS = [4, 4, 4]
DEADLINE = 150.0
START = time.time()

def _elapsed():
    return time.time() - START

def _form_add(a, b):
    d = dict(a)
    for k, v in b.items():
        d[k] = d.get(k, 0) + v
    return {k: v for k, v in d.items() if v != 0}

def _form_sub(a, b):
    d = dict(a)
    for k, v in b.items():
        d[k] = d.get(k, 0) - v
    return {k: v for k, v in d.items() if v != 0}

def _badd(A, B):
    return [[_form_add(A[i][j], B[i][j]) for j in range(2)] for i in range(2)]

def _bsub(A, B):
    return [[_form_sub(A[i][j], B[i][j]) for j in range(2)] for i in range(2)]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    # A blocks
    A11 = [[{ (0,0):1 },{ (0,1):1 }],
           [ { (1,0):1 },{ (1,1):1 }]]
    A12 = [[{ (0,2):1 },{ (0,3):1 }],
           [ { (1,2):1 },{ (1,3):1 }]]
    A21 = [[{ (2,0):1 },{ (2,1):1 }],
           [ { (3,0):1 },{ (3,1):1 }]]
    A22 = [[{ (2,2):1 },{ (2,3):1 }],
           [ { (3,2):1 },{ (3,3):1 }]]
    # B blocks
    B11 = [[{ (0,0):1 },{ (0,1):1 }],
           [ { (1,0):1 },{ (1,1):1 }]]
    B12 = [[{ (0,2):1 },{ (0,3):1 }],
           [ { (1,2):1 },{ (1,3):1 }]]
    B21 = [[{ (2,0):1 },{ (2,1):1 }],
           [ { (3,0):1 },{ (3,1):1 }]]
    B22 = [[{ (2,2):1 },{ (2,3):1 }],
           [ { (3,2):1 },{ (3,3):1 }]]

    # outer Strassen 7 products (X block, Y block)
    outer_products = [
        (_badd(A11, A22), _badd(B11, B22)),   # M1
        (_badd(A21, A22), B11),                # M2
        (A11, _bsub(B12, B22)),                # M3
        (A22, _bsub(B21, B11)),                # M4
        (_badd(A11, A12), B22),                # M5
        (_bsub(A21, A11), _badd(B11, B12)),    # M6
        (_bsub(A12, A22), _badd(B21, B22)),    # M7
    ]
    # outer target: list of (row_off, col_off, sign)
    outer_target = [
        [(0,0,1)],                       # M1 -> C11
        [(2,0,1)],                       # M2 -> C21
        [(0,2,1)],                       # M3 -> C12
        [(0,0,-1),(2,0,1)],              # M4 -> C11(-), C21(+)
        [(0,2,1)],                       # M5 -> C12
        [(0,0,1)],                       # M6 -> C11
        [(2,2,1)],                       # M7 -> C22
    ]
    # inner Strassen: for Z = X*Y, 7 products (U_form, V_form, contrib)
    # contrib: list of (ri, ci, coeff)
    def inner_products(X, Y):
        x00,x01 = X[0][0], X[0][1]
        x10,x11 = X[1][0], X[1][1]
        y00,y01 = Y[0][0], Y[0][1]
        y10,y11 = Y[1][0], Y[1][1]
        return [
            (_form_add(x00,x11), _form_add(y00,y11), [(0,0,1),(1,1,1)]),     # P1
            (_form_add(x10,x11), y00, [(1,0,1),(1,1,-1)]),                  # P2
            (x00, _form_sub(y01,y11), [(0,1,1)]),                           # P3
            (x11, _form_sub(y10,y00), [(0,0,-1),(1,0,1)]),                  # P4
            (_form_add(x00,x01), y11, [(0,1,1)]),                           # P5
            (_form_sub(x10,x00), _form_add(y00,y01), [(0,0,1)]),            # P6
            (_form_sub(x01,x11), _form_add(y10,y11), [(1,1,1)]),            # P7
        ]

    r = 0
    for t in range(7):
        X, Y = outer_products[t]
        for (uform, vform, contrib) in inner_products(X, Y):
            # build W form
            wform = {}
            for (ri, ci, ccoef) in contrib:
                for (ro, co, s) in outer_target[t]:
                    i = ro + ri
                    k = co + ci
                    wform[(i,k)] = wform.get((i,k), 0) + s * ccoef
            # fill U, V, W
            for (i, j), c in uform.items():
                U[i*m + j][r] += c
            for (j, k), c in vform.items():
                V[j*p + k][r] += c
            for (i, k), c in wform.items():
                W[k*n + i][r] += c
            r += 1

    # verify tensor identity
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for rr in range(rank):
                    s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                if s != 1:
                    ok = False
                    break
            if not ok: break
        if not ok: break
    if not ok:
        # fallback to schoolbook
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
    return {"U": U, "V": V, "W": W}
