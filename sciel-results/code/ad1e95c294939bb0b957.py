import time
import numpy as np

def _get_s7():
    u = np.zeros((4, 7))
    v = np.zeros((4, 7))
    w = np.zeros((4, 7))
    u[[0,3], 0] = 1.0
    v[[0,3], 0] = 1.0
    w[[0,3], 0] = 1.0
    u[[2,3], 1] = 1.0
    v[0, 1] = 1.0
    w[1, 1] = 1.0
    w[3, 1] = -1.0
    u[0, 2] = 1.0
    v[[1,3], 2] = [1.0, -1.0]
    w[[2,3], 2] = 1.0
    u[3, 3] = 1.0
    v[[2,0], 3] = [1.0, -1.0]
    w[[0,1], 3] = 1.0
    u[[0,1], 4] = 1.0
    v[3, 4] = 1.0
    w[0, 4] = -1.0
    w[2, 4] = 1.0
    u[[2,0], 5] = [1.0, -1.0]
    v[[0,1], 5] = 1.0
    w[3, 5] = 1.0
    u[[1,3], 6] = [1.0, -1.0]
    v[[2,3], 6] = 1.0
    w[0, 6] = 1.0
    return u, v, w

def _get_t3():
    a = np.eye(3)
    b = np.eye(3)
    c = np.eye(3)
    return a, b, c

def _assemble(n, m, p):
    su, sv, sw = _get_s7()
    tu, tv, tw = _get_t3()
    total = 42
    U = np.zeros((8, total))
    V = np.zeros((24, total))
    W = np.zeros((12, total))
    for blk in range(2):
        base = blk * 21
        for a in range(7):
            for b in range(3):
                idx = base + a * 3 + b
                for ii in range(2):
                    for jj in range(2):
                        gc = blk * 2 + jj
                        cv = su[ii * 2 + jj, a] * tu[0, b]
                        if cv:
                            U[ii * 4 + gc, idx] += cv
                for jj in range(2):
                    gc = blk * 2 + jj
                    for kk1 in range(2):
                        for kk2 in range(3):
                            gk = kk1 * 3 + kk2
                            cv = sv[jj * 2 + kk1, a] * tv[kk2, b]
                            if cv:
                                V[gc * 6 + gk, idx] += cv
                for kk1 in range(2):
                    for kk2 in range(3):
                        gk = kk1 * 3 + kk2
                        for ii in range(2):
                            cv = sw[kk1 * 2 + ii, a] * tw[kk2, b]
                            if cv:
                                W[gk * 2 + ii, idx] += cv
    return U, V, W

def _check(U, V, W):
    r = U.shape[1]
    for i in range(2):
        for j in range(4):
            for k in range(6):
                acc = 0.0
                for s in range(r):
                    acc += U[i*4+j, s] * V[j*6+k, s] * W[k*2+i, s]
                if abs(acc - 1.0) > 1e-9:
                    return False
    return True

def construct():
    t0 = time.time()
    U, V, W = _assemble(2, 4, 6)
    if _check(U, V, W):
        out = {}
        for key, mat in [('U', U), ('V', V), ('W', W)]:
            out[key] = []
            for row in mat:
                out[key].append([int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row])
        return out
    if time.time() - t0 < 110:
        rank = 48
        Uf = np.zeros((8, rank))
        Vf = np.zeros((24, rank))
        Wf = np.zeros((12, rank))
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(6):
                    Uf[i*4+j, col] = 1.0
                    Vf[j*6+k, col] = 1.0
                    Wf[k*2+i, col] = 1.0
                    col += 1
        out = {}
        for key, mat in [('U', Uf), ('V', Vf), ('W', Wf)]:
            out[key] = []
            for row in mat:
                out[key].append([int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row])
        return out
    return {"U": [], "V": [], "W": []}
