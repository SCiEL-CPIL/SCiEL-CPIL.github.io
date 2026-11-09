"""Optimized 12-stage product formula for 6-site XXZ Heisenberg chain."""
import time
import numpy as np
from scipy.linalg import expm

_start = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill


def _build_system():
    """Build A (odd bonds + half field) and B (even bonds + half field)."""
    n = 6
    dim = 2 ** n
    delta = 0.8
    h = 0.3

    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])

    def kron_all(ops):
        r = np.array([[1.0]])
        for op in ops:
            r = np.kron(r, op)
        return r

    def site_op(i, op):
        ops = [np.eye(2)] * n
        ops[i] = op
        return kron_all(ops)

    def bond(i):
        # XXZ bond between site i and i+1
        return (0.5 * (site_op(i, sx) @ site_op(i + 1, sx)
                       + site_op(i, sy) @ site_op(i + 1, sy))
                + delta * site_op(i, sz) @ site_op(i + 1, sz))

    field = h * sum(site_op(j, sz) for j in range(n))

    # Odd bonds: (0,1),(2,3),(4,5); Even bonds: (1,2),(3,4)
    A = sum(bond(i) for i in (0, 2, 4)) + 0.5 * field
    B = sum(bond(i) for i in (1, 3)) + 0.5 * field
    return A, B


def _eval(a, b, A, B, times):
    """Mean spectral-norm error of the product formula vs exact propagator."""
    total = 0.0
    for t in times:
        U = np.eye(2 ** 6)
        for k in range(len(a)):
            U = expm(-1j * b[k] * t * B) @ U
            U = expm(-1j * a[k] * t * A) @ U
        Ue = expm(-1j * t * (A + B))
        total += np.linalg.norm(U - Ue, 2)
    return total / len(times)


def _project(a, b):
    """Clip to bounds and renormalize each list to sum 1."""
    a = np.clip(np.asarray(a, dtype=float), -50.0, 50.0)
    b = np.clip(np.asarray(b, dtype=float), -50.0, 50.0)
    sa = a.sum()
    sb = b.sum()
    if abs(sa) > 1e-12:
        a = a / sa
    else:
        a = np.full_like(a, 1.0 / len(a))
    if abs(sb) > 1e-12:
        b = b / sb
    else:
        b = np.full_like(b, 1.0 / len(b))
    return a, b


def _seed_strang():
    """Second-order Strang coefficients as the starting point."""
    a = np.full(12, 1.0 / 12.0)
    b = np.full(12, 1.0 / 12.0)
    return a, b


def coefficients(stages):
    """Return (a, b), two lists of 12 finite numbers in [-50,50], each summing to 1."""
    n = 12
    times = [0.25, 0.5, 1.0]
    A, B = _build_system()

    best_a, best_b = _seed_strang()
    best_err = _eval(best_a, best_b, A, B, times)

    # L-BFGS-B over the 24 free coefficients, projected back to constraints.
    from scipy.optimize import minimize

    def obj(x):
        if time.time() > _DEADLINE - 5.0:
            return best_err
        a, b = _project(x[:n], x[n:])
        return _eval(a, b, A, B, times)

    x0 = np.concatenate([best_a, best_b])
    best_x = x0.copy()

    try:
        res = minimize(obj, x0, method="L-BFGS-B",
                       options={"maxiter": 60, "maxfun": 300,
                                "ftol": 1e-10, "gtol": 1e-7})
        if time.time() < _DEADLINE - 5.0:
            a, b = _project(res.x[:n], res.x[n:])
            e = _eval(a, b, A, B, times)
            if e < best_err:
                best_err = e
                best_a, best_b = a, b
                best_x = res.x.copy()
    except Exception:
        pass

    # Local polish: Nelder-Mead on a small neighborhood of the best point.
    if time.time() < _DEADLINE - 10.0:
        try:
            from scipy.optimize import minimize

            def obj2(x):
                if time.time() > _DEADLINE - 5.0:
                    return best_err
                a, b = _project(x[:n], x[n:])
                return _eval(a, b, A, B, times)

            r2 = minimize(obj2, best_x, method="Nelder-Mead",
                          options={"maxiter": 400, "xatol": 1e-8,
                                   "fatol": 1e-9, "adaptive": True})
            a, b = _project(r2.x[:n], r2.x[n:])
            e = _eval(a, b, A, B, times)
            if e < best_err:
                best_err = e
                best_a, best_b = a, b
        except Exception:
            pass

    # Final safety check: guarantee validity and sum.
    best_a, best_b = _project(best_a, best_b)
    best_a = np.clip(best_a, -50.0, 50.0)
    best_b = np.clip(best_b, -50.0, 50.0)
    if abs(best_a.sum()) > 1e-12:
        best_a = best_a / best_a.sum()
    if abs(best_b.sum()) > 1e-12:
        best_b = best_b / best_b.sum()

    return [float(v) for v in best_a], [float(v) for v in best_b]
