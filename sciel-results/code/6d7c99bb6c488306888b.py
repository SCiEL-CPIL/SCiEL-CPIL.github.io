"""Optimized product-formula coefficients for six-site Heisenberg chain splitting."""
import time
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

_DEADLINE = 140.0
_T0 = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field).

    6-site open XXZ chain, delta=0.8, field h=0.3.
    H = sum_{j=0..4} [ Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + delta Sz_j Sz_{j+1} ]
        - h sum_j Sz_j
    Bonds (0-indexed): 0:(0,1),1:(1,2),2:(2,3),3:(3,4),4:(4,5)
    Even bonds: 0,2,4  -> B
    Odd bonds:  1,3    -> A
    Half field added to each.
    """
    delta = 0.8
    h = 0.3
    n = 6
    dim = 2 ** n
    I = np.eye(2)
    Sx = 0.5 * np.array([[0, 1], [1, 0]])
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]])
    Sz = 0.5 * np.array([[1, 0], [0, -1]])

    def kron_all(ops):
        out = np.array([[1.0]])
        for op in ops:
            out = np.kron(out, op)
        return out

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    for j in range(n - 1):
        opsA = [I] * n
        opsB = [I] * n
        opsA[j] = Sx; opsA[j + 1] = Sx
        opsB[j] = Sx; opsB[j + 1] = Sx
        A += kron_all(opsA)
        B += kron_all(opsB)
        opsA = [I] * n; opsB = [I] * n
        opsA[j] = Sy; opsA[j + 1] = Sy
        opsB[j] = Sy; opsB[j + 1] = Sy
        A += kron_all(opsA)
        B += kron_all(opsB)
        opsA = [I] * n; opsB = [I] * n
        opsA[j] = Sz; opsA[j + 1] = Sz
        opsB[j] = Sz; opsB[j + 1] = Sz
        A += delta * kron_all(opsA)
        B += delta * kron_all(opsB)
    for j in range(n):
        opsA = [I] * n; opsB = [I] * n
        opsA[j] = Sz; opsB[j] = Sz
        A -= (h / 2.0) * kron_all(opsA)
        B -= (h / 2.0) * kron_all(opsB)
    return A, B


def _make_objective(A, B, Uexact):
    def objective(x):
        x = np.asarray(x, dtype=float)
        sa = float(np.sum(x[:12])); sb = float(np.sum(x[12:]))
        a = x[:12] + (1.0 - sa) / 12.0
        b = x[12:] + (1.0 - sb) / 12.0
        total = 0.0
        for t in (0.25, 0.5, 1.0):
            U = np.eye(64, dtype=complex)
            for k in range(12):
                if b[k] != 0.0:
                    U = expm(-1j * b[k] * t * B) @ U
                if a[k] != 0.0:
                    U = expm(-1j * a[k] * t * A) @ U
            total += np.linalg.norm(U - Uexact[t], 2)
        return total / 3.0
    return objective


def coefficients(stages):
    stages = int(stages)
    if stages < 1:
        stages = 12
    A, B = _build_hamiltonians()
    Uexact = {0.25: expm(-1j * 0.25 * (A + B)),
              0.5: expm(-1j * 0.5 * (A + B)),
              1.0: expm(-1j * 1.0 * (A + B))}
    obj = _make_objective(A, B, Uexact)

    def consistent(x):
        x = np.asarray(x, dtype=float)
        a = x[:stages] + (1.0 - np.sum(x[:stages])) / stages
        b = x[stages:] + (1.0 - np.sum(x[stages:])) / stages
        return a, b

    # Baseline: uniform (Strang-like)
    base_a = [1.0 / stages] * stages
    base_b = [1.0 / stages] * stages
    base_x = np.array(base_a + base_b, dtype=float)
    base_score = obj(base_x)
    best_x = base_x.copy()
    best_score = base_score

    # Seed starts
    rng = np.random.default_rng(12345)
    starts = []
    # Strang-style: a half at ends, b full interior pattern
    s_a = [0.5 / (stages // 2)] * stages
    s_b = [0.0] * stages
    for i in range(stages // 2):
        s_b[2 * i] = 1.0 / (stages // 2)
    starts.append(np.array(s_a + s_b, dtype=float))
    # symmetric random starts
    for _ in range(6):
        starts.append(np.concatenate([
            rng.normal(1.0 / stages, 0.15, stages),
            rng.normal(1.0 / stages, 0.15, stages)]))

    options = {"maxiter": 80, "ftol": 1e-9, "gtol": 1e-7, "maxfun": 200}
    for x0 in starts:
        if _remaining() < 12.0:
            break
        try:
            res = minimize(obj, x0, method="L-BFGS-B", options=options)
        except Exception:
            continue
        if res.x is not None and np.all(np.isfinite(res.x)):
            s = obj(res.x)
            if s < best_score - 1e-12:
                best_score = s
                best_x = res.x.copy()

    a, b = consistent(best_x)
    # clamp to [-50,50]
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    # final consistency after clipping
    a = a + (1.0 - np.sum(a)) / stages
    b = b + (1.0 - np.sum(b)) / stages
    return [float(v) for v in a], [float(v) for v in b]
