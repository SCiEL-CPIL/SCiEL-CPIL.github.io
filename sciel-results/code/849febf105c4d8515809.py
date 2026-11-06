"""Model-and-optimize: parameterize the 3-layer Ry/CZ circuit and directly
minimize the exact mean energy gap per spin against the exact ground state."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.optimize

# Explicit global wall-clock deadline guard (evaluator kills at 150 s).
_GLOBAL_DEADLINE = time.time() + 142.0
_CACHE = {}


def _build_hamiltonian(n, coupling, field):
    """Build the full Hamiltonian matrix and return (H, E0)."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    # ZZ terms
    for i in range(n - 1):
        ma = 1 << i
        mb = 1 << (i + 1)
        for state in range(dim):
            za = 1.0 if ((state & ma) == 0) else -1.0
            zb = 1.0 if ((state & mb) == 0) else -1.0
            H[state, state] += -coupling * za * zb
    # X terms
    for i in range(n):
        m = 1 << i
        for state in range(dim):
            H[state, state ^ m] += -field
    H = (H + H.conj().T) / 2.0
    evals = np.linalg.eigvalsh(H)
    E0 = float(evals[0])
    return H, E0


_GATE_CACHE = {}

def _get_ry_indices(n, qubit):
    """Pre-compute index arrays for Ry on qubit."""
    key = ('ry', n, qubit)
    if key not in _GATE_CACHE:
        dim = 2 ** n
        mask = 1 << qubit
        idx = np.arange(dim)
        zero_mask = (idx & mask) == 0
        zero_idx = idx[zero_mask]
        one_idx = zero_idx | mask
        _GATE_CACHE[key] = (zero_idx, one_idx)
    return _GATE_CACHE[key]

def _get_cz_indices(n, a, b):
    """Pre-compute index array for CZ on qubits (a, b)."""
    key = ('cz', n, a, b)
    if key not in _GATE_CACHE:
        dim = 2 ** n
        mask = (1 << a) | (1 << b)
        idx = np.arange(dim)
        both_one = (idx & mask) == mask
        _GATE_CACHE[key] = np.where(both_one)[0]
    return _GATE_CACHE[key]

def _circuit_state(theta, n, layers):
    """Simulate the circuit and return the state vector."""
    dim = 2 ** n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t * 0.5)
            s = np.sin(t * 0.5)
            zero_idx, one_idx = _get_ry_indices(n, i)
            psi_new = psi.copy()
            psi_new[zero_idx] = c * psi[zero_idx] - s * psi[one_idx]
            psi_new[one_idx] = s * psi[zero_idx] + c * psi[one_idx]
            psi = psi_new
        if l < layers - 1:
            for i in range(n - 1):
                both_one_idx = _get_cz_indices(n, i, i + 1)
                psi[both_one_idx] = -psi[both_one_idx]
    return psi


def _energy_of_state(psi, H):
    return float(np.real(np.vdot(psi, H @ psi)))


def _optimize_case(n, coupling, field, layers, time_budget):
    H, E0 = _build_hamiltonian(n, coupling, field)
    dim = 2 ** n
    n_params = layers * n

    def obj(theta_flat):
        theta = theta_flat.reshape(layers, n)
        psi = _circuit_state(theta, n, layers)
        return (_energy_of_state(psi, H) - E0) / n

    bounds = [(-np.pi, np.pi)] * n_params
    rng = np.random.default_rng(42)

    # Build diverse starting points
    starts = [np.zeros((layers, n))]
    n_starts = 40
    for _ in range(n_starts):
        starts.append(rng.uniform(-np.pi, np.pi, size=(layers, n)))

    best_theta = np.zeros((layers, n))
    best_val = np.inf
    t0 = time.time()

    for s in starts:
        if time.time() - t0 > time_budget - 3.0:
            break
        try:
            res = scipy.optimize.minimize(
                obj, s.reshape(-1), method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-11},
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass

    # Polish pass
    remaining = time_budget - (time.time() - t0)
    if remaining > 2.0 and best_val < np.inf:
        try:
            res = scipy.optimize.minimize(
                obj, best_theta.reshape(-1), method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-13},
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass

    # Second polish if time permits
    remaining = time_budget - (time.time() - t0)
    if remaining > 2.0 and best_val < np.inf:
        try:
            res = scipy.optimize.minimize(
                obj, best_theta.reshape(-1), method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 300, "ftol": 1e-15, "gtol": 1e-14},
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass

    # Third high-precision polish if time permits
    remaining = time_budget - (time.time() - t0)
    if remaining > 1.5 and best_val < np.inf:
        try:
            res = scipy.optimize.minimize(
                obj, best_theta.reshape(-1), method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 200, "ftol": 1e-16, "gtol": 1e-15},
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass

    if not np.all(np.isfinite(best_theta)):
        best_theta = np.zeros((layers, n))
    return best_theta


def angles(n, coupling, field, layers):
    key = (int(n), float(coupling), float(field), int(layers))
    if key in _CACHE:
        return _CACHE[key].copy()

    remaining = _GLOBAL_DEADLINE - time.time()
    if remaining <= 5.0:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        _CACHE[key] = theta
        return theta.copy()

    budget = min(remaining * 0.88, 40.0)

    try:
        theta = _optimize_case(n, coupling, field, layers, budget)
    except Exception:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2

    if not np.all(np.isfinite(theta)):
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2

    _CACHE[key] = theta
    return theta.copy()
# EVOLVE-BLOCK-END
