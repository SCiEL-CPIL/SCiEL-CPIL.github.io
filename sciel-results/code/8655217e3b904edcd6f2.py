"""Optimized product-formula coefficients for Heisenberg-chain time evolution."""
import numpy as np
import time
from scipy.optimize import minimize

_AB_cache = None

def _build_AB():
    N = 6
    anisotropy = 0.8
    field = 0.3
    dim = 2 ** N
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    for site in range(N):
        ops = [I] * N
        ops[site] = sz
        A += 0.5 * field * kron_all(ops)
        ops = [I] * N
        ops[site] = sz
        B += 0.5 * field * kron_all(ops)

    for i, j in [(0, 1), (2, 3), (4, 5)]:
        ops = [I] * N; ops[i] = sx; ops[j] = sx
        A += 0.5 * anisotropy * kron_all(ops)
        ops = [I] * N; ops[i] = sy; ops[j] = sy
        A += 0.5 * anisotropy * kron_all(ops)
        ops = [I] * N; ops[i] = sx; ops[j] = sx
        A += 0.5 * kron_all(ops)
        ops = [I] * N; ops[i] = sy; ops[j] = sy
        A += 0.5 * kron_all(ops)

    for i, j in [(1, 2), (3, 4)]:
        ops = [I] * N; ops[i] = sx; ops[j] = sx
        B += 0.5 * anisotropy * kron_all(ops)
        ops = [I] * N; ops[i] = sy; ops[j] = sy
        B += 0.5 * anisotropy * kron_all(ops)
        ops = [I] * N; ops[i] = sx; ops[j] = sx
        B += 0.5 * kron_all(ops)
        ops = [I] * N; ops[i] = sy; ops[j] = sy
        B += 0.5 * kron_all(ops)

    return A, B

def _get_AB():
    global _AB_cache
    if _AB_cache is None:
        _AB_cache = _build_AB()
    return _AB_cache

def _objective(coeffs):
    a = np.array(coeffs[:12], dtype=float)
    b = np.array(coeffs[12:24], dtype=float)
    A, B = _get_AB()
    dim = A.shape[0]
    H = A + B
    times = [0.25, 0.5, 1.0]
    total_err = 0.0
    for t in times:
        U_exact = np.linalg.matrix_exp(-1j * t * H)
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            U = np.linalg.matrix_exp(-1j * b[k] * t * B) @ U
            U = np.linalg.matrix_exp(-1j * a[k] * t * A) @ U
        total_err += np.linalg.norm(U - U_exact, 2)
    return total_err / len(times)

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 50.0

    x0 = np.ones(24) / 12.0

    best = x0.copy()
    best_val = _objective(x0)

    result = minimize(_objective, x0, method='L-BFGS-B',
                      bounds=[(-50, 50)] * 24,
                      options={'maxiter': 300, 'maxfun': 8000, 'ftol': 1e-15, 'gtol': 1e-12})
    if result.fun < best_val and time.time() < deadline:
        best = result.x
        best_val = result.fun

    a = best[:12].copy()
    b = best[12:24].copy()
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    sa, sb = np.sum(a), np.sum(b)
    if sa != 0:
        a = a * (1.0 / sa)
    if sb != 0:
        b = b * (1.0 / sb)
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    sa, sb = np.sum(a), np.sum(b)
    if sa != 0:
        a = a * (1.0 / sa)
    if sb != 0:
        b = b * (1.0 / sb)

    return list(a), list(b)
