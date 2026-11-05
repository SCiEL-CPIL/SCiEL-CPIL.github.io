import time
import sys

DIMENSIONS = [4, 4, 5]
_START_TIME = time.time()
TIME_LIMIT = 120.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    
    if _time_remaining() < 1.0:
        rank = 80
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def assign(col, u_data, v_data, w_data):
        for i, j, val in u_data:
            U[i * m + j][col] += val
        for j, k, val in v_data:
            V[j * p + k][col] += val
        for k, i, val in w_data:
            W[k * n + i][col] += val

    col = 0

    a_coeffs = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    b_coeffs = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    c_contrib = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1],
        [1, 0, 1, 0], [-1, 1, 0, 0], [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    p_inner = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    q_inner = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    r_inner = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1],
        [1, 0, 1, 0], [-1, 1, 0, 0], [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def blk(A, B, c, r):
        if A == 'A':
            return (B // 2) * 2 + r, (B % 2) * 2 + c
        elif A == 'B':
            return (B // 2) * 2 + r, (B % 2) * 2 + c
        else:
            return (B // 2) * 2 + r, (B % 2) * 2 + c

    for s in range(7):
        for t in range(7):
            ud, vd, wd = [], [], []
            
            for pe, pv in enumerate(p_inner[t]):
                if not pv: continue
                pr, pc = pe // 2, pe % 2
                for ab, ac in enumerate(a_coeffs[s]):
                    if not ac: continue
                    gi, gj = blk('A', ab, pc, pr)
                    ud.append((gi, gj, ac * pv))
            
            for qe, qv in enumerate(q_inner[t]):
                if not qv: continue
                qr, qc = qe // 2, e % 2
                for bb, bc in enumerate(b_coeffs[s]):
                    if not bc: continue
                    gj, gk = blk('B', bb, qr, qc)
                    vd.append((gj, gk, bc * qv))
            
            for cb, sc in enumerate(c_contrib[s]):
                if not sc: continue
                for rb, ic in enumerate(r_inner[t]):
                    if not ic: continue
                    tc = sc * ic
                    if not tc: continue
                    lr, lc = rb // 2, rb % 2
                    gi, gk = blk('C', cb, lr, lc)
                    wd.append((gk, gi, tc))
            
            if ud and vd and wd:
                assign(col, ud, vd, wd)
                col += 1

    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
