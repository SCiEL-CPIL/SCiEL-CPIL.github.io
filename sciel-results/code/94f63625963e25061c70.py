"""Optimized 12-stage product formula for 6-site XXZ Heisenberg chain time evolution.

Structural change from seed: instead of a fixed Strang splitting, build the exact
A (odd bonds + half field) and B (even bonds + half field) Hamiltonians and run a
direct local optimization (L-BFGS-B with numerical gradients, plus a Nelder-Mead
polish) to minimize the mean spectral-norm error at t = 0.25, 0.5, 1.0.  A hard
wall-clock deadline guard returns the best valid result found.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize, rosenbrock  # rosenbrock import kept minimal; not used
from scipy.linalg import expm

# ---------- deadline guard ----------
_START = time.time()
_DEADLINE = 110.0  # safety margin below 120s wall-clock kill

def _time_left():
    return _DEADLINE - (time.time() - _START)

# ---------- physics: build A and B ----------
def _build_hamiltonians():
    """Return (A, B) for a 6-site open XXZ chain, anisotropy=0.8, field=0.3.

    A = sum over odd bonds (i,i+1) for i in {0,2,4} + half uniform field
    B = sum over even bonds (i,i+1) for i in {1,3,5} + half uniform field
    """
    N = 6
    dim = 2 ** N
    anisotropy = 0.8
    field = 0.3

    # Pauli matrices
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    ident = np.eye(2, dtype=complex)

    def kron_all(ops):
        """Kronecker product of a list of 2x2 matrices in site order 0..N-1."""
        result = np.array([[1.0]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    def bond_term(i, j):
        """XXZ bond term on sites i, j: 0.5*(sx_i sx_j + sy_i sy_j) + anisotropy * sz_i sz_j."""
        ops = [ident] * N
        ops[i] = sx
        ops[j] = sx
        term_xx = kron_all(ops)
        ops[i] = sy
        ops[j] = sy
        term_yy = kron_all(ops)
        ops[i] = sz
        ops[j] = sz
        term_zz = kron_all(ops)
        return 0.5 * (term_xx + term_yy) + anisotropy * term_zz

    # Half uniform field: 0.5 * field * sum_i sz_i
    half_field = 0.5 * field
    field_op = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        ops = [ident] * N
        ops[i] = sz
        field_op += kron_all(ops)
    half_field_op = half_field * field_op

    # A: odd bonds (0,1),(2,3),(4,5) + half field
    A = np.zeros((dim, dim), dtype=complex)
    for i in range(0, N, 2):
        A += bond_term(i, i + 1)
    A += half_field_op

    # B: even bonds (1,2),(3,4) + half field  (bond (5,6) doesn't exist for N=6 open chain)
    B = np.zeros((dim, dim), dtype=complex)
    for i in range(1, N - 1, 2):
        B += bond_term(i, i + 1)
    B += half_field_op

    return A, B

# ---------- objective ----------
def _objective(coeffs, A, B, exact_exps, times):
    """Mean spectral-norm error at the three evaluation times."""
    a = coeffs[:12]
    b = coeffs[12:24]
    total = 0.0
    for idx, t in enumerate(times):
        U = np.eye(64, dtype=complex)
        for k in range(12):
            ak = a[k]
            bk = b[k]
            # exp(-i bk t B)
            if abs(bk) > 1e-15:
                U = expm(-1j * bk * t * B) @ U
            # exp(-i ak t A)
            if abs(ak) > 1e-15:
                U = expm(-1j * ak * t * A) @ U
        err = np.linalg.norm(U - exact_exps[idx], 2)
        total += err
    return total / len(times)

# ---------- main ----------
def coefficients(stages):
    """Return (a, b): two lists of 12 finite numbers in [-50,50] with sum 1 each."""
    global _START
    if not hasattr(coefficients, '_cached'):
        _START = time.time()
        coefficients._cached = True

    times = [0.25, 0.5, 1.0]
    A, B = _build_hamiltonians()
    exact_exps = [expm(-1j * t * (A + B)) for t in times]

    # Initial guess: Strang splitting
    a0 = [1.0 / 12.0] * 12
    b0 = [1.0 / 12.0] * 12
    x0 = np.array(a0 + b0, dtype=float)

    bounds = [(-50.0, 50.0)] * 24

    best_x = x0.copy()
    best_err = _objective(best_x, A, B, exact_exps, times)

    # Random restarts with L-BFGS-B
    rng = np.random.default_rng(42)
    n_restarts = 6

    for r in range(n_restarts):
        if _time_left() < 20.0:
            break
        # Perturb around the Strang guess
        if r == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + rng.uniform(-0.1, 0.1, size=24)
            # Enforce bounds
            x_start = np.clip(x_start, -50.0, 50.0)

        try:
            res = minimize(
                _objective, x_start, args=(A, B, exact_exps, times),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500}
            )
            if res.success or res.fun < best_err:
                x_candidate = np.clip(res.x, -50.0, 50.0)
                err = _objective(x_candidate, A, B, exact_exps, times)
                if err < best_err:
                    best_err = err
                    best_x = x_candidate
        except Exception:
            pass

    # Nelder-Mead polish on the best L-BFGS-B result
    if _time_left() > 15.0:
        try:
            res_nm = minimize(
                _objective, best_x, args=(A, B, exact_exps, times),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            x_nm = np.clip(res_nm.x, -50.0, 50.0)
            err_nm = _objective(x_nm, A, B, exact_exps, times)
            if err_nm < best_err:
                best_err = err_nm
                best_x = x_nm
        except Exception:
            pass

    a = list(best_x[:12])
    b = list(best_x[12:24])

    # Enforce consistency: sum(a) = sum(b) = 1
    # If the optimizer drifted, rescale (but keep within bounds)
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-12:
        a = [x / sa for x in a]
    else:
        a = [1.0 / 12.0] * 12
    if abs(sb) > 1e-12:
        b = [x / sb for x in b]
    else:
        b = [1.0 / 12.0] * 12

    # Clamp to [-50, 50] and re-normalize if clamping broke the sum
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-12:
        a = [x / sa for x in a]
    if abs(sb) > 1e-12:
        b = [x / sb for x in b]

    # Final safety: ensure finite
    a = [float(x) if math.isfinite(x) else 1.0/12.0 for x in a]
    b = [float(x) if math.isfinite(x) else 1.0/12.0 for x in b]

    return a, b
