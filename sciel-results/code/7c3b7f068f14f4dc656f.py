"""Optimized product-formula coefficients for a 6-site open XXZ Heisenberg chain.

Strategy:
  1. Build A (odd bonds + half field) and B (even bonds + half field) exactly as the
     evaluator would (128x128 Hermitian matrices).
  2. Precompute exp(-i t (A+B)) for t in {0.25, 0.5, 1.0}.
  3. Use scipy L-BFGS-B to minimize the mean spectral-norm error over the 24
     coefficients, subject to sum(a)=sum(b)=1 and bounds [-50,50].
  4. A hard deadline guard returns the best feasible solution found before ~140 s.
"""

import time
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before the 150 s kill

# ---- Build the chain matrices (done once, cached) -------------------------
_A = None
_B = None
_EXACT = None  # list of exp(-i t (A+B)) for t in {0.25,0.5,1.0}
_TIMES = (0.25, 0.5, 1.0)


def _build():
    """Build A, B and the exact propagators for the six-site open XXZ chain."""
    global _A, _B, _EXACT
    if _A is not None:
        return
    dim = 6
    anis = 0.8
    field = 0.3
    n = 2 ** dim
    I2 = np.eye(2)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = np.array([[0.0, -1j], [1j, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])

    def kron_all(ops):
        r = np.array([[1.0 + 0j]])
        for op in ops:
            r = np.kron(r, op)
        return r

    A = np.zeros((n, n), dtype=complex)
    B = np.zeros((n, n), dtype=complex)

    # Half of the uniform field goes into each half
    field_half = 0.5 * field
    for s in range(dim):
        ops = [I2] * dim
        ops[s] = sz
        A += field_half * kron_all(ops)
        B += field_half * kron_all(ops)

    # Bond operators: (1/2)(sigma_i^x sigma_j^x + sigma_i^y sigma_j^y + anis sigma_i^z sigma_j^z)
    for bond in range(dim - 1):
        i, j = bond, bond + 1
        for comp, op in (("x", sx), ("y", sy), ("z", sz)):
            ops = [I2] * dim
            ops[i] = op
            ops[j] = op
            term = kron_all(ops)
            coeff = 0.5 * anis if comp == "z" else 0.5
            if bond % 2 == 0:
                B += coeff * term
            else:
                A += coeff * term

    _A = A
    _B = B
    _EXACT = [expm(-1j * t * (A + B)) for t in _TIMES]


def _objective(x):
    """Mean spectral-norm error for coefficient vector x = [a_1..a_12, b_1..b_12]."""
    a = x[:12]
    b = x[12:]
    err = 0.0
    for idx, t in enumerate(_TIMES):
        U = np.eye(128, dtype=complex)
        for k in range(12):
            bk = b[k]
            if bk != 0.0:
                U = expm(-1j * bk * t * _B) @ U
            ak = a[k]
            if ak != 0.0:
                U = expm(-1j * ak * t * _A) @ U
        err += np.linalg.norm(U - _EXACT[idx], 2)
    return err / len(_TIMES)


def _objective_and_grad(x):
    """Objective with a numerical (central) gradient for L-BFGS-B."""
    f = _objective(x)
    g = np.zeros_like(x)
    h = 1e-5
    for i in range(len(x)):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (_objective(xp) - _objective(xm)) / (2.0 * h)
    return f, g


def coefficients(stages):
    """Return (a, b): two lists of `stages` finite numbers in [-50,50] with sum 1 each."""
    _build()
    n = stages  # 12

    # Feasible starting point: uniform coefficients (sum = 1)
    x0 = np.concatenate([np.ones(n) / n, np.ones(n) / n])

    bounds = [(-50.0, 50.0)] * (2 * n)
    # Equality constraints sum(a)=1, sum(b)=1
    cons = [
        {"type": "eq", "fun": lambda x: np.sum(x[:n]) - 1.0,
         "jac": lambda x: np.concatenate([np.ones(n), np.zeros(n)])},
        {"type": "eq", "fun": lambda x: np.sum(x[n:]) - 1.0,
         "jac": lambda x: np.concatenate([np.zeros(n), np.ones(n)])},
    ]

    best_x = x0.copy()
    best_f = _objective(x0)

    def _deadline():
        return time.time() > _DEADLINE

    # Primary optimization: L-BFGS-B with numerical gradient
    try:
        if not _deadline():
            res = minimize(_objective_and_grad, x0, method="L-BFGS-B",
                           jac=True, bounds=bounds, constraints=cons,
                           options={"maxiter": 300, "maxfun": 3000,
                                    "gtol": 1e-6, "ftol": 1e-12})
            if res.success or res.fun < best_f:
                if np.all(np.isfinite(res.x)) and np.all(res.x >= -50) and np.all(res.x <= 50):
                    best_x = res.x
                    best_f = res.fun
    except Exception:
        pass

    # Secondary: a few random restarts if time remains
    rng = np.random.default_rng(42)
    for _ in range(5):
        if _deadline():
            break
        # Small random perturbation around the current best
        x_try = best_x + rng.normal(0.0, 0.05, size=2 * n)
        # Project back to satisfy sum constraints approximately
        x_try[:n] += (1.0 - np.sum(x_try[:n])) / n
        x_try[n:] += (1.0 - np.sum(x_try[n:])) / n
        x_try = np.clip(x_try, -50.0, 50.0)
        try:
            res2 = minimize(_objective_and_grad, x_try, method="L-BFGS-B",
                            jac=True, bounds=bounds, constraints=cons,
                            options={"maxiter": 100, "maxfun": 1000,
                                     "gtol": 1e-5, "ftol": 1e-10})
            if res2.fun < best_f:
                if np.all(np.isfinite(res2.x)) and np.all(res2.x >= -50) and np.all(res2.x <= 50):
                    best_x = res2.x
                    best_f = res2.fun
        except Exception:
            pass

    a = list(best_x[:n])
    b = list(best_x[n:])

    # Final safety: enforce sum = 1 and bounds
    sa = sum(a)
    if sa != 0.0:
        a = [v / sa for v in a]
    sb = sum(b)
    if sb != 0.0:
        b = [v / sb for v in b]
    a = [max(-50.0, min(50.0, v)) for v in a]
    b = [max(-50.0, min(50.0, v)) for v in b]

    # Re-normalize after clipping
    sa = sum(a)
    if sa != 0.0:
        a = [v / sa for v in a]
    sb = sum(b)
    if sb != 0.0:
        b = [v / sb for v in b]

    return a, b
