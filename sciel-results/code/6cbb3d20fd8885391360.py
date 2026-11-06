"""Optimize 3-layer Ry/CZ circuit for open transverse-field Ising chains."""
import numpy as np
import time
from scipy.optimize import minimize

_H_cache = {}
_e0_cache = {}
_idx_cache = {}


def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # Z terms: -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    # X terms: -field * sum X_i
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
    e0 = float(eigvals[0])
    _e0_cache[key] = e0
    return e0


def _get_indices(n):
    """Pre-compute index arrays for each qubit's 0 and 1 subspaces, and CZ masks."""
    key = n
    if key in _idx_cache:
        return _idx_cache[key]
    dim = 1 << n
    qubit_indices = []
    for q in range(n):
        mask = 1 << q
        idx_0 = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.int64)
        idx_1 = idx_0 + mask
        qubit_indices.append((idx_0, idx_1))
    # Pre-compute CZ masks for adjacent pairs
    cz_masks = []
    all_idx = np.arange(dim)
    for i in range(n - 1):
        m1 = 1 << i
        m2 = 1 << (i + 1)
        both_set = ((all_idx & m1) != 0) & ((all_idx & m2) != 0)
        cz_masks.append(both_set)
    _idx_cache[n] = (qubit_indices, cz_masks)
    return _idx_cache[n]


def _circuit_state(n, thetas, layers):
    """
    Vectorized application of the Ry/CZ circuit.
    thetas: (layers, n) array of rotation angles.
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.float64)
    state[0] = 1.0

    qubit_indices, cz_masks = _get_indices(n)

    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)

            idx_0, idx_1 = qubit_indices[q]
            
            vals_0 = state[idx_0]
            vals_1 = state[idx_1]
            
            new_vals_0 = c * vals_0 - s * vals_1
            new_vals_1 = s * vals_0 + c * vals_1
            
            state[idx_0] = new_vals_0
            state[idx_1] = new_vals_1

        # Apply CZ between layers (except after the last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                state[cz_masks[i]] *= -1.0

    return state


def _objective(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    psi = _circuit_state(n, thetas, layers)
    H = _build_H(n, coupling, field)
    e_circuit = float(psi @ H @ psi)
    return (e_circuit - e0) / n


def _mean_field_init(n, coupling, field, layers):
    theta = np.zeros((layers, n), dtype=np.float64)
    # Heuristic initialization
    theta[0] = field / 4.0
    theta[1] = field / 3.0
    theta[-1] = field / 2.0
    return theta


def _product_state_init(n, coupling, field, layers):
    theta = np.zeros((layers, n), dtype=np.float64)
    theta[-1] = np.pi / 2
    theta[1] = np.pi / 4
    theta[0] = np.pi / 8
    return theta


def _transfer_matrix_init(n, coupling, field, layers, rng):
    """
    MPS-inspired initialization using exact ground state expectations.
    """
    H = _build_H(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    ground_state = eigvecs[:, 0].real
    ground_state = ground_state / np.linalg.norm(ground_state)

    theta = np.zeros((layers, n), dtype=np.float64)
    dim = 1 << n

    for q in range(n):
        z_q = 0.0
        x_q = 0.0
        for idx in range(dim):
            bit_q = (idx >> q) & 1
            z_val = 1.0 if bit_q == 0 else -1.0
            z_q += z_val * ground_state[idx] * ground_state[idx]
            idx_flip = idx ^ (1 << q)
            x_q += ground_state[idx] * ground_state[idx_flip]

        theta_eff = np.arctan2(x_q, z_q)
        theta[0, q] = theta_eff * 0.2
        theta[1, q] = theta_eff * 0.3
        theta[2, q] = theta_eff * 0.5

    return theta


def angles(n, coupling, field, layers):
    """Main entry point: optimize Ry angles for the given parameters."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    e0 = _exact_ground_energy(n, coupling, field)
    
    # Generate diverse initial guesses
    rng = np.random.RandomState(42)
    inits = []
    inits.append(_mean_field_init(n, coupling, field, layers))
    inits.append(_product_state_init(n, coupling, field, layers))
    inits.append(_transfer_matrix_init(n, coupling, field, layers, rng))
    for _ in range(4):
        inits.append(rng.uniform(-np.pi, np.pi, size=(layers, n)))

    best_theta = None
    best_energy = np.inf

    # Phase 1: L-BFGS-B from each initial guess
    for init in inits:
        if time.time() > deadline - 5.0:
            break

        params0 = init.flatten()
        try:
            res = minimize(
                _objective,
                params0,
                args=(n, coupling, field, layers, e0),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass

    # Phase 2: Polish with Nelder-Mead from best
    if best_theta is not None and time.time() < deadline - 10.0:
        try:
            res2 = minimize(
                _objective,
                best_theta.flatten(),
                args=(n, coupling, field, layers, e0),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if res2.fun < best_energy:
                best_energy = res2.fun
                best_theta = res2.x.reshape((layers, n))
        except Exception:
            pass

    # Phase 3: Additional L-BFGS-B polish
    if best_theta is not None and time.time() < deadline - 5.0:
        try:
            res3 = minimize(
                _objective,
                best_theta.flatten(),
                args=(n, coupling, field, layers, e0),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res3.fun < best_energy:
                best_energy = res3.fun
                best_theta = res3.x.reshape((layers, n))
        except Exception:
            pass

    # Fallback: if optimization failed, return a reasonable default
    if best_theta is None:
        best_theta = _product_state_init(n, coupling, field, layers)

    return best_theta.astype(np.float64)
