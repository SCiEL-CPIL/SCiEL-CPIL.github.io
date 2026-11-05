"""Product-formula design: fit 12-stage coefficients to the exact Heisenberg-chain propagator."""
import time
import numpy as np
from scipy.linalg import expm

_DEADLINE = 140.0
_T_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _T_START)


def _build_hamiltonian():
    """Six-site open XXZ Heisenberg chain, anisotropy 0.8, field 0.3, split into odd/even bonds."""
    N = 6
    dim = 2 ** N
    aniso = 0.8
    field = 0.3
    Sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    Sy = np.array([[0.0, -1j], [1j, 0.0]])
    Sz = np.array([[1.0, 0.0], [0.0, -1.0]])
    I2 = np.eye(2)

    def op(k, M):
        out = M
        for j in range(N):
            if j == k:
                out = np.kron(out, M)
            else:
                out = np.kron(out, I2)
        return out

    H = np.zeros((dim, dim), dtype=complex)
    for j in range(N - 1):
        H += aniso * op(j, Sz) @ op(j + 1, Sz)
        H += 0.5 * (op(j, Sx) @ op(j + 1, Sx) + op(j, Sy) @ op(j + 1, Sy))
    for j in range(N):
        H += field * op(j, Sz)

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    for j in range(N - 1):
        bond = aniso * op(j, Sz) @ op(j + 1, Sz) + 0.5 * (
            op(j, Sx) @ op(j + 1, Sx) + op(j, Sy) @ op(j + 1, Sy))
        if j % 2 == 0:
            B += bond
        else:
            A += bond
    for j in range(N):
        hf = 0.5 * field * op(j, Sz)
        if j % 2 == 0:
            B += hf
        else:
            A += hf
    return A, B


def _apply(a, b, A, B, t):
    U = np.eye(A.shape[0], dtype=complex)
    for k in range(len(a)):
        U = expm(-1j * b[k] * t * B) @ U
        U = expm(-1j * a[k] * t * A) @ U
    return U


def _loss(x, A, B, exact):
    a = x[:12]
    b = x[12:24]
    s = 0.0
    for i, t in enumerate((0.25, 0.5, 1.0)):
        U = _apply(a, b, A, B, t)
        d = U - exact[i]
        s += float(np.linalg.norm(d, 2))
    return s / 3.0


def _clamp(x):
    x = np.clip(x, -50.0, 50.0)
    x[:12] -= x[:12].sum() - 1.0
    x[12:24] -= x[12:24].sum() - 1.0
    return np.clip(x, -50.0, 50.0)


def coefficients(stages):
    stages = int(stages)
    if stages != 12:
        # Fallback: uniform Strang-style if not 12.
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b

    A, B = _build_hamiltonian()
    exact = [expm(-1j * t * (A + B)) for t in (0.25, 0.5, 1.0)]

    def _valid(x):
        return (np.all(np.isfinite(x)) and abs(x[:12].sum() - 1.0) < 1e-6
                and abs(x[12:24].sum() - 1.0) < 1e-6
                and np.all(x >= -50.0) and np.all(x <= 50.0))

    # Initial guess: equal-weight symmetric-ish split.
    x0 = np.array([1.0 / 12] * 12 + [1.0 / 12] * 12)
    x_best = x0.copy()
    best = _loss(x_best, A, B, exact)

    # Local optimization (L-BFGS-B) with periodic deadline checks.
    try:
        from scipy.optimize import minimize
        res = minimize(_loss, x_best, method='L-BFGS-B',
                       options={'maxiter': 300, 'maxfun': 400, 'disp': False})
        if _valid(res.x):
            lc = _clamp(res.x)
            l = _loss(lc, A, B, exact)
            if l < best:
                best, x_best = l, lc
    except Exception:
        pass

    if _remaining() < 12.0:
        a, b = x_best[:12], x_best[12:24]
        return [float(v) for v in a], [float(v) for v in b]

    # Nelder-Mead refinement on the clamped space.
    try:
        from scipy.optimize import minimize
        x_nm = _clamp(x_best.copy())
        res2 = minimize(lambda z: _loss(_clamp(z), A, B, exact), x_nm,
                        method='Nelder-Mead',
                        options={'maxiter': 400, 'maxfun': 600, 'disp': False})
        if _valid(res2.x):
            lc = _clamp(res2.x)
            l = _loss(lc, A, B, exact)
            if l < best:
                best, x_best = l, lc
    except Exception:
        pass

    a, b = x_best[:12], x_best[12:24]
    return [float(v) for v in a], [float(v) for v in b]
