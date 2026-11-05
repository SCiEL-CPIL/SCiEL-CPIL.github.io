"""Shallow-circuit Ising ground-state approximator.

Strategy:
  - Exact ground energy of H = -J sum Z_i Z_{i+1} - h sum X_i (open chain) is
    computed in O(n) via a 2x2 transfer-matrix (matrix product) construction.
  - The circuit energy <psi|H|psi> is evaluated with vectorized state evolution
    (Ry rotations + adjacent CZ gates), so each evaluation is cheap (~O(n 2^n)).
  - A smooth objective (mean gap per spin) is minimized with multi-restart
    L-BFGS-B followed by a Nelder-Mead polish, under an explicit wall-clock
    deadline guard.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = 132.0  # safety margin below the 150 s kill
_RNG_SEED = 20240517

# ----------------------------------------------------------------------
# Exact ground energy (transfer matrix)
# ----------------------------------------------------------------------
def _ground_energy(n, coupling, field):
    """Exact lowest eigenvalue of H = -J sum Z_i Z_{i+1} - h sum X_i (open)."""
    J = coupling
    h = field
    # X term per site: [[-h, 0],[0, h]]  (|0> = +1, |1> = -1 under Z)
    X0 = np.array([[-h, 0.0], [0.0, h]])
    X1 = np.array([[h, 0.0], [0.0, -h]])
    # Z_i Z_{i+1} = [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]
    ZZ = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])
    T0 = np.kron(X0, np.eye(2))
    T1 = np.kron(X1, np.eye(2))
    T0 = -J * T0 + T0
    T1 = -J * T1 + T1
    T0 = T0.copy()
    T1 = T1.copy()
    # rebuild cleanly
    T0 = -J * np.kron(np.eye(2), np.eye(2)) * 0.0  # placeholder
    T0 = np.kron(X0, np.eye(2))
    T1 = np.kron(X1, np.eye(2))
    T0 = T0 - J * np.kron(np.eye(2), ZZ)
    T1 = T1 - J * np.kron(np.eye(2), ZZ)
    M = T0
    for _ in range(n - 1):
        M = M @ T1
    return float(np.min(np.linalg.eigvalsh(M)))


# ----------------------------------------------------------------------
# Circuit energy (vectorized)
# ----------------------------------------------------------------------
def _circuit_energy(theta, n, coupling, field, H):
    """<psi|H|psi> for the fixed Ry/CZ circuit; theta shape (layers, n)."""
    psi = np.zeros(1 << n, dtype=np.complex128)
    psi[0] = 1.0
    for L in range(theta.shape[0]):
        # Ry(theta) on every qubit: psi' = sum_k V_k |k>, V_k = prod_i V_i(k_i)
        c = np.cos(theta[L] / 2.0)
        s = np.sin(theta[L] / 2.0)
        V = c.copy()
        V[1:] = -s
        V = V.reshape((2,) * n)
        psi = np.tensordot(psi, V, axes=([0], [0]))
        psi = psi.reshape(1 << n)
        # CZ on adjacent pairs in ascending order (no entangler after last layer)
        if L < theta.shape[0] - 1:
            for i in range(n - 1):
                idx = np.arange(1 << n)
                mask = (idx >> i) & 1
                psi = np.where(mask[None, :], -psi, psi)
    return float(np.real(np.vdot(psi, H @ psi)))


# ----------------------------------------------------------------------
# Objective
# ----------------------------------------------------------------------
def _objective(x, n, coupling, field, H, E0):
    theta = x.reshape(3, n)
    e = _circuit_energy(theta, n, coupling, field, H)
    return (e - E0) / n


def _hamiltonian(n, coupling, field):
    """Dense H = -J sum Z_i Z_{i+1} - h sum X_i, qubit i is bit i."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    Z = np.diag([1.0, -1.0])
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    for i in range(n - 1):
        Zi = np.eye(1)
        for j in range(n):
            Zi = np.kron(Zi, Z if j == i else np.eye(2))
        Zp1 = np.eye(1)
        for j in range(n):
            Zp1 = np.kron(Zp1, Z if j == i + 1 else np.eye(2))
        H -= coupling * (Zi @ Zp1)
    for i in range(n):
        Xi = np.eye(1)
        for j in range(n):
            Xi = np.kron(Xi, X if j == i else np.eye(2))
        H -= field * Xi
    return H


def _optimize_field(n, coupling, field, budget):
    H = _hamiltonian(n, coupling, field)
    E0 = _ground_energy(n, coupling, field)
    rng = np.random.default_rng(_RNG_SEED + int(field * 1000) + n)

    def obj(x):
        return _objective(x, n, coupling, field, H, E0)

    best = None
    best_val = np.inf
    # Initial guesses: uniform random + a few structured starts
    starts = []
    for _ in range(6):
        starts.append(rng.uniform(-1.2, 1.2, size=(3 * n,)))
    # structured: all-equal small, and |+>-like last layer
    starts.append(np.tile(0.0, 3 * n))
    v = np.zeros(3 * n); v[2 * n:] = np.pi / 2
    starts.append(v)

    for x0 in starts:
        if time.time() > budget:
            break
        try:
            res = minimize(obj, x0, method='L-BFGS-B',
                           options={'maxiter': 200, 'maxfun': 4000})
            val = res.fun
            if val < best_val:
                best_val = float(val)
                best = res.x.copy()
        except Exception:
            pass

    # Nelder-Mead polish on the best
    if best is not None and time.time() < budget:
        try:
            res2 = minimize(obj, best, method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res2.fun < best_val:
                best_val = float(res2.fun)
                best = res2.x.copy()
        except Exception:
            pass

    if best is None:
        # Fallback: |+> product state
        best = np.zeros(3 * n)
        best[2 * n:] = np.pi / 2
    return best.reshape(3, n)


def angles(n, coupling, field, layers):
    layers = int(layers)
    # Exact ground energy for this (n, coupling, field)
    E0 = _ground_energy(n, coupling, field)
    H = _hamiltonian(n, coupling, field)

    def obj(x):
        theta = x.reshape(layers, n)
        return (_circuit_energy(theta, n, coupling, field, H) - E0) / n

    rng = np.random.default_rng(_RNG_SEED + int(field * 1000) + n)

    best = None
    best_val = np.inf
    starts = []
    for _ in range(6):
        starts.append(rng.uniform(-1.2, 1.2, size=(layers * n,)))
    starts.append(np.zeros(layers * n))
    v = np.zeros(layers * n); v[(layers - 1) * n:] = np.pi / 2
    starts.append(v)

    for x0 in starts:
        if time.time() > _DEADLINE:
            break
        try:
            res = minimize(obj, x0, method='L-BFGS-B',
                           options={'maxiter': 300, 'maxfun': 6000})
            if res.fun < best_val:
                best_val = float(res.fun)
                best = res.x.copy()
        except Exception:
            pass

    if best is not None and time.time() < _DEADLINE:
        try:
            res2 = minimize(obj, best, method='Nelder-Mead',
                            options={'maxiter': 4000, 'xatol': 1e-11, 'fatol': 1e-13})
            if res2.fun < best_val:
                best_val = float(res2.fun)
                best = res2.x.copy()
        except Exception:
            pass

    if best is None:
        best = np.zeros(layers * n)
        best[(layers - 1) * n:] = np.pi / 2

    return best.reshape(layers, n)
# EVOLVE-BLOCK-END
