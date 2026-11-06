"""Combined: cached circuit ops + multi-restart L-BFGS-B with polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_H_cache = {}
_e0_cache = {}
_cz_mask_cache = {}
_idx_cache = {}


def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # Z terms
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    # X terms
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    _H_cache[key] = H
    return H


def _exact_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    H = _build_H(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0]
    _e0_cache[key] = e0
    return e0


def _get_cz_masks(n):
    key = n
    if key in _cz_mask_cache:
        return _cz_mask_cache[key]
    dim = 1 << n
    arange = np.arange(dim)
    masks = []
    for i in range(n - 1):
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        both_set = ((arange & mask1) != 0) & ((arange & mask2) != 0)
        masks.append(both_set)
    _cz_mask_cache[key] = masks
    return masks


def _get_idx_pairs(n):
    key = n
    if key in _idx_cache:
        return _idx_cache[key]
    dim = 1 << n
    idx_0 = {}
    idx_1 = {}
    for q in range(n):
        mask = 1 << q
        idx_0[q] = np.array([i for i in range(dim) if (i & mask) == 0])
        idx_1[q] = idx_0[q] + mask
    _idx_cache[key] = (idx_0, idx_1)
    return (idx_0, idx_1)


def _circuit_state_fast(n, thetas, layers):
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j

    idx_0, idx_1 = _get_idx_pairs(n)
    cz_masks = _get_cz_masks(n)

    for layer in range(layers):
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)

            i0 = idx_0[q]
            i1 = idx_1[q]

            vals_0 = state[i0]
            vals_1 = state[i1]

            state[i0] = c * vals_0 - s * vals_1
            state[i1] = s * vals_0 + c * vals_1

        if layer < layers - 1:
            for m in cz_masks:
                state[m] *= -1.0

    return state


def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state_fast(n, thetas, layers)
    return np.real(np.conj(psi) @ H @ psi)


def _objective(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0

    e0 = _exact_ground_energy(n, coupling, field)

    best_x = np.zeros(layers * n)
    best_val = _objective(best_x, n, coupling, field, layers, e0)

    # Build diverse initializations
    inits = []
    inits.append(np.zeros(layers * n))
    inits.append(np.full(layers * n, np.pi / 2))
    inits.append(np.full(layers * n, np.pi))
    inits.append(np.full(layers * n, -np.pi / 2))

    ratio = field / coupling if coupling != 0 else field
    theta_est = (np.pi / 2) * (1 - np.exp(-ratio))
    inits.append(np.full(layers * n, theta_est))

    # Fixed heuristic angles based on physics
    inits.append(np.full(layers * n, 0.5))
    inits.append(np.full(layers * n, 1.0))
    inits.append(np.full(layers * n, 1.5))

    # Layer-wise variation
    t1 = np.full(layers * n, 0.3)
    t1[n:2*n] = 0.8
    t1[2*n:3*n] = 0.5
    inits.append(t1)

    t2 = np.full(layers * n, 0.7)
    t2[n:2*n] = 0.4
    t2[2*n:3*n] = 0.9
    inits.append(t2)

    # Random inits
    for i in range(12):
        np.random.seed(100 + i)
        inits.append(np.random.uniform(-np.pi, np.pi, size=(layers * n)))

    for i in range(8):
        np.random.seed(200 + i)
        inits.append(np.random.uniform(0, np.pi, size=(layers * n)))

    # Limit restarts based on n
    max_restarts = 30 if n <= 4 else 24
    num_restarts = min(len(inits), max_restarts)

    for i in range(num_restarts):
        if time.time() > deadline:
            break

        x_init = inits[i].copy()

        try:
            res = minimize(lambda p: _objective(p, n, coupling, field, layers, e0),
                          x_init, method='L-BFGS-B',
                          options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Polish phase
    remaining = deadline - time.time()
    if remaining > 5.0:
        try:
            res = minimize(lambda p: _objective(p, n, coupling, field, layers, e0),
                          best_x, method='L-BFGS-B',
                          options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Second polish from a slightly perturbed best
    remaining = deadline - time.time()
    if remaining > 5.0:
        try:
            np.random.seed(42)
            perturbed = best_x + np.random.normal(0, 0.1, size=(layers * n))
            res = minimize(lambda p: _objective(p, n, coupling, field, layers, e0),
                          perturbed, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    return best_x.reshape((layers, n))
# EVOLVE-BLOCK-END
