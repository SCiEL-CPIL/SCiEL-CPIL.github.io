import time

DIMENSIONS = [3, 3, 4]

_DEADLINE = 120.0
_START = None

def _remaining():
    global _START
    if _START is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank34():
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal c
        if c >= rank:
            return
        U[u_idx][c] += u_val
        V[v_idx][c] += v_val
        W[w_idx][c] += w_val
        c += 1

    def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal c
        U[a00][c] = 1; U[a11][c] = 1
        V[b00][c] = 1; V[b11][c] = 1
        W[c00][c] = 1; W[c11][c] = 1
        c += 1
        U[a10][c] = 1; U[a11][c] = 1
        V[b00][c] = 1
        W[c10][c] = 1; W[c11][c] = -1
        c += 1
        U[a00][c] = 1
        V[b01][c] = 1; V[b11][c] = -1
        W[c01][c] = 1; W[c11][c] = 1
        c += 1
        U[a11][c] = 1
        V[b10][c] = 1; V[b00][c] = -1
        W[c00][c] = 1; W[c10][c] = 1
        c += 1
        U[a00][c] = 1; U[a01][c] = 1
        V[b11][c] = 1
        W[c00][c] = -1; W[c01][c] = 1
        c += 1
        U[a10][c] = 1; U[a00][c] = -1
        V[b00][c] = 1; V[b01][c] = 1
        W[c11][c] = 1
        c += 1
        U[a01][c] = 1; U[a11][c] = -1
        V[b10][c] = 1; V[b11][c] = 1
        W[c00][c] = 1
        c += 1

    def c_idx(i, k): return k * n + i
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k

    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1), 
                 v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1), 
                 c_idx(0,0), c_idx(0,1), c_idx(1,0), c_idx(1,1))

    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1), 
                 v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3), 
                 c_idx(0,2), c_idx(0,3), c_idx(1,2), c_idx(1,3))

    add_term(u_idx(0,2), 1, v_idx(2,0), 1, c_idx(0,0), 1)
    add_term(u_idx(0,2), 1, v_idx(2,1), 1, c_idx(0,1), 1)
    add_term(u_idx(1,2), 1, v_idx(2,0), 1, c_idx(1,0), 1)
    add_term(u_idx(1,2), 1, v_idx(2,1), 1, c_idx(1,1), 1)

    add_term(u_idx(0,2), 1, v_idx(2,2), 1, c_idx(0,2), 1)
    add_term(u_idx(0,2), 1, v_idx(2,3), 1, c_idx(0,3), 1)
    add_term(u_idx(1,2), 1, v_idx(2,2), 1, c_idx(1,2), 1)
    add_term(u_idx(1,2), 1, v_idx(2,3), 1, c_idx(1,3), 1)

    add_term(u_idx(2,0), 1, v_idx(0,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,0), 1, v_idx(0,1), 1, c_idx(2,1), 1)
    add_term(u_idx(2,1), 1, v_idx(1,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,1), 1, v_idx(1,1), 1, c_idx(2,1), 1)

    add_term(u_idx(2,0), 1, v_idx(0,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,0), 1, v_idx(0,3), 1, c_idx(2,3), 1)
    add_term(u_idx(2,1), 1, v_idx(1,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,1), 1, v_idx(1,3), 1, c_idx(2,3), 1)

    add_term(u_idx(2,2), 1, v_idx(2,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,2), 1, v_idx(2,1), 1, c_idx(2,1), 1)

    add_term(u_idx(2,2), 1, v_idx(2,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,2), 1, v_idx(2,3), 1, c_idx(2,3), 1)

    return {"U": U, "V": V, "W": W}

def construct():
    global _START
    _START = time.time()
    try:
        return _construct_rank34()
    except:
        return _schoolbook()
