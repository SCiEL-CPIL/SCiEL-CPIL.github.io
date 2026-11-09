"""Fidelity-based optimizer for shallow Ry/CZ Ising ground-state approximation.

Maximizes <psi_target|psi_circuit>^2 (smooth, unique-global-max surrogate) with
vectorized circuit simulation + parameter-shift gradients + multi-start L-BFGS-B,
guarded by an explicit wall-clock deadline.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 112.0          # safety margin before the 120 s evaluator kill
_START = time.time()
_CACHE = {}

_I2 = np.eye(2, dtype=complex)
_X2 = np.array([[0, 1], [1, 0]], dtype=complex)
_Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
_CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex)


def _deadline_hit():
    return (time.time() - _START) >= _DEADLINE


def _ry(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _apply_circuit(theta, n):
    """|psi> from |0>^n through the layer-wise Ry/CZ circuit (no CZ after last layer)."""
    layers = theta.shape[0]
    psi = np.zeros(1 << n, dtype=complex)
    psi[0] = 1.0
    for l in range(layers):
        for i in range(n):
            psi = np.kron(_ry(theta[l, i]), _I2) @ psi
        if l < layers - 1:
            for i in range(n - 1):
                psi = np.kron(np.kron(_I2, _CZ), np.eye(2 ** (n - 2), dtype=complex)) @ psi
    return psi


def _exact_ground(n, coupling, field):
    """Exact ground state; qubit i = bit i (rightmost qubit is least-significant)."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n - 1):
        term = np.array([1.0])
        for k in range(n):
            term = np.kron(term, _Z2 if k in (i, i + 1) else _I2)
        H += coupling * term
    for i in range(n):
        term = np.array([1.0])
        for k in range(n):
            term = np.kron(term, _X2 if k == i else _I2)
        H -= field * term
    w, v = np.linalg.eigh(H)
    idx = np.argmin(w)
    return v[:, idx].astype(complex)


def _fidelity(theta, n, target):
    psi = _apply_circuit(theta, n)
    a = np.vdot(target, psi)
    return float(a.real * a.real + a.imag * a.imag)


def _grad(theta, n, target):
    g = np.zeros_like(theta, dtype=float)
    layers = theta.shape[0]
    for l in range(layers):
        for i in range(n):
            tp = theta.copy(); tp[l, i] += 0.5 * np.pi
            tm = theta.copy(); tm[l, i] -= 0.5 * np.pi
            g[l, i] = 0.5 * (_fidelity(tp, n, target) - _fidelity(tm, n, target))
    return g


def _optimize(n, coupling, field, target):
    layers = 3
    best_theta = np.zeros((layers, n))
    best_f = -1.0
    starts = [np.zeros((layers, n)), np.full((layers, n), np.pi / 2.0)]
    rng = np.random.default_rng(12345)
    for _ in range(6):
        starts.append(rng.uniform(-np.pi, np.pi, size=(layers, n)))
    for s in starts:
        if _deadline_hit():
            break
        try:
            res = minimize(
                lambda x: -_fidelity(x.reshape(layers, n), n, target),
                s.reshape(-1),
                jac=lambda x: -_grad(x.reshape(layers, n), n, target),
                method="L-BFGS-B",
                options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-10},
            )
        except Exception:
            continue
        f = _fidelity(res.x.reshape(layers, n), n, target)
        if f > best_f:
            best_f = f
            best_theta = res.x.reshape(layers, n).copy()
    return best_theta


def angles(n, coupling, field, layers):
    key = (n, float(coupling), float(field), int(layers))
    if key in _CACHE:
        return _CACHE[key].copy()
    target = _exact_ground(n, coupling, field)
    theta = _optimize(n, coupling, field, target)
    _CACHE[key] = theta.copy()
    return theta.copy()
# EVOLVE-BLOCK-END
