import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 135.0   # safety margin vs 150s kill
_N = 6
_DELTA = 0.8
_H = 0.3
_SIGMA = np.array([[0, 1], [1, 0]])
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def _time_left():
    return _DEADLINE - time.time()


def _matrices():
    A = np.zeros((_N, _N), dtype=complex)
    B = np.zeros((_N, _N), dtype=complex)
    for i in range(_N - 1):
        S = np.kron(np.eye(i), np.kron(_SIGMA, np.eye(_N - i - 1)))
        if i % 2 == 0:
            A += S
        else:
            B += S
    field = np.kron(_SZ, np.eye(_N - 1))
    A += 0.5 * _H * field
    B += 0.5 * _H * field
    return A, B


def _prod(U, a, b):
    for k in range(len(a)):
        U = np.dot(np.exp(-1j * b[k] * B), U)
        U = np.dot(np.exp(-1j * a[k] * A), U)
    return U


def _true_err(a, b):
    e = 0.0
    for t in (0.25, 0.5, 1.0):
        U = _prod(np.eye(_N, dtype=complex), a, b)
        e += np.linalg.norm(U - np.exp(-1j * t * (A + B)), 2)
    return e / 3.0


def _local_err(a, b):
    # order-2 local error: sum_k (1/2 a_k^2 - 1/2 b_k^2 + a_k b_k) * [A, B]
    s = 0.0
    for k in range(len(a)):
        s += 0.5 * a[k] * a[k] - 0.5 * b[k] * b[k] + a[k] * b[k]
    C = np.dot(A, B) - np.dot(B, A)
    return abs(s) * np.linalg.norm(C, 2)


def _nm(f, x0, iters, step):
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        p = list(x0)
        p[i] = x0[i] + step
        simplex.append(p)
    vals = [f(p) for p in simplex]
    for _ in range(iters):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if _time_left() < 5.0:
            break
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        xr = [centroid[j] + (centroid[j] - simplex[0][j]) for j in range(n)]
        fr = f(xr)
        if fr < vals[0]:
            xe = [centroid[j] + 2 * (centroid[j] - simplex[0][j]) for j in range(n)]
            fe = f(xe)
            if fe < fr:
                simplex[0], vals[0] = xe, fe
            else:
                simplex[0], vals[0] = xr, fr
        elif fr < vals[-1]:
            simplex[0], vals[0] = xr, fr
        else:
            xc = [centroid[j] + 0.5 * (simplex[0][j] - centroid[j]) for j in range(n)]
            fc = f(xc)
            if fc < vals[0]:
                simplex[0], vals[0] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    vals[i] = f(simplex[i])
    order = sorted(range(n + 1), key=lambda i: vals[i])
    return simplex[order[0]], vals[order[0]]


def coefficients(stages):
    global A, B
    A, B = _matrices()
    stages = int(stages)
    if stages < 2:
        stages = 2

    def unpack(v):
        a = list(v[:stages])
        b = list(v[stages:])
        sa = sum(a)
        sb = sum(b)
        if sa != 0:
            a = [x / sa for x in a]
        if sb != 0:
            b = [x / sb for x in b]
        return a, b

    def obj(v):
        a, b = unpack(v)
        return _local_err(a, b)

    # symmetric base: a=(1/2,1/2) per substep, b=(1,0)
    sub = stages // 2
    base = [0.0] * (2 * stages)
    for k in range(sub):
        base[2 * k] = 0.5
        base[2 * k + 1] = 0.5
        base[stages + 2 * k] = 1.0
        base[stages + 2 * k + 1] = 0.0
    while len(base) < 2 * stages:
        base.append(0.0)
    base = base[:2 * stages]

    best_a, best_b = unpack(base)
    best_true = _true_err(best_a, best_b)

    iters = 200
    while _time_left() > 30.0:
        x0 = list(base)
        for j in range(2 * stages):
            x0[j] += np.random.uniform(-0.25, 0.25)
        x, _ = _nm(obj, x0, iters, 0.15)
        a, b = unpack(x)
        if _time_left() < 15.0:
            break
        e = _true_err(a, b)
        if e < best_true:
            best_true = e
            best_a, best_b = a, b
            base = list(x)
            iters = max(80, iters // 2)
    return best_a, best_b


# precompute matrices at import so first call is fast
A, B = _matrices()
