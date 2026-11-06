# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = 142.0  # safety margin under the 150s wall-clock limit


def _pauli(n, i):
    """Return kron product of single-qubit Pauli matrices, qubit i is bit i (rightmost)."""
    mats = []
    for k in range(n):
        if (i >> k) & 1:
            mats.append(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64))
        else:
            mats.append(np.eye(2, dtype=np.float64))
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _build_hamiltonian(n, coupling, field):
    """Open transverse-field Ising chain H = -J sum Z_i Z_{i+1} - g sum X_i.
    Qubit i is bit i, so |psi[a]> has qubit i = bit i of a."""
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    I = np.eye(2, dtype=np.float64)
    dim = 1 << n

    def kron_all(mats):
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    H = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        mats = [I] * n
        mats[i] = Z
        mats[i + 1] = Z
        H -= coupling * kron_all(mats)
    for i in range(n):
        mats = [I] * n
        mats[i] = X
        H -= field * kron_all(mats)
    return H


def _exact_ground(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    w, V = np.linalg.eigh(H)
    return w[0], V[:, 0].astype(np.float64)


def _simulate(theta, n, layers):
    """Start |0>^n; each layer Ry(theta) on every qubit; CZ on adjacent pairs ascending
    between layers (none after the final layer). Returns float64 statevector."""
    dim = 1 << n
    psi = np.zeros(dim, dtype=np.float64)
    psi[0] = 1.0
    for L in range(layers):
        c = np.cos(0.5 * theta[L])
        s = np.sin(0.5 * theta[L])
        for i in range(n):
            step = 1 << i
            block = step * 2
            for base in range(0, dim, block):
                for a in range(step):
                    lo = base + a
                    hi = lo + step
                    psi[lo], psi[hi] = c * psi[lo] - s * psi[hi], s * psi[lo] + c * psi[hi]
        if L < layers - 1:
            for i in range(n - 1):
                step = 1 << i
                block = step * 2
                for base in range(0, dim, block):
                    for a in range(step):
                        hi = base + a + step
                        psi[hi] = -psi[hi]
    return psi


def _objective(params, n, layers, gvec):
    psi = _simulate(params.reshape(layers, n), n, layers)
    return -np.dot(psi, gvec)


def angles(n, coupling, field, layers):
    dim = 1 << n
    g0, gvec = _exact_ground(n, coupling, field)
    total = layers * n

    def time_left():
        return _DEADLINE - (time.time() - _T0)

    def try_nm(x0, maxiter, tol):
        if time_left() < 1.0:
            return None
        res = minimize(_objective, x0, args=(n, layers, gvec), method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol})
        if res.x is not None and np.all(np.isfinite(res.x)):
            return res.x
        return None

    candidates = []

    # 1. Uniform per-qubit product ansatz: |psi> = prod_i (c_i|0> + s_i|1>),
    #    c_i = cos(theta_i), s_i = sin(theta_i) (Ry(theta)|0> = c|0> + s|1>).
    #    Match amplitudes of the exact ground state to get a strong initial guess.
    g = gvec.reshape([2] * n)
    prod_theta = np.zeros(n)
    for i in range(n):
        ax = np.take(g, [0, 1], axis=i)
        c0 = np.sum(ax * ax)
        c1 = np.sum(ax * ax) if False else 0.0
        c0 = float(np.sum(np.take(g, [0], axis=i) ** 2))
        c1 = float(np.sum(np.take(g, [1], axis=i) ** 2))
        prod_theta[i] = np.arctan2(np.sqrt(c1), np.sqrt(c0))
    for L in range(layers - 1):
        candidates.append(np.tile(prod_theta, (1, n)).reshape(-1))
    candidates.append(np.tile(prod_theta, (1, n)).reshape(-1))
    candidates.append(np.full(total, np.pi / 2))
    candidates.append(np.zeros(total))

    # 2. Random starts.
    rng = np.random.default_rng(12345)
    n_random = 20
    for _ in range(n_random):
        candidates.append(rng.uniform(-np.pi, np.pi, size=total))

    best = None
    best_val = np.inf
    for x0 in candidates:
        if time_left() < 2.0:
            break
        res = try_nm(x0, 3000, 1e-13)
        if res is None:
            continue
        val = _objective(res, n, layers, gvec)
        if val < best_val:
            best_val = val
            best = res

    # Polish the best with tighter tolerances while time remains.
    if best is not None:
        polish_iters = 0
        while time_left() > 3.0 and polish_iters < 6:
            res = try_nm(best, 2000, 1e-15)
            if res is None:
                break
            val = _objective(res, n, layers, gvec)
            if val < best_val - 1e-15:
                best_val = val
                best = res
            polish_iters += 1

    if best is None:
        best = np.zeros(total)

    return best.reshape(layers, n)
# EVOLVE-BLOCK-END
