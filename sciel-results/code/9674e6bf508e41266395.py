"""Optimize shallow circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


def _apply_ry(state, n, qubit, theta):
    """Apply Ry(theta) to qubit on state of n qubits (little-endian)."""
    dim = 1 << n
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    new_state = np.zeros_like(state)
    # Vectorized: indices where qubit bit is 0
    mask = np.arange(dim)
    bit_zero = (mask >> qubit) & 1 == 0
    bit_one = ~bit_zero
    idx0 = mask[bit_zero]
    idx1 = mask[bit_one]
    new_state[idx0] = c * state[idx0] - s * state[idx1]
    new_state[idx1] = s * state[idx0] + c * state[idx1]
    return new_state


def _apply_cz(state, n, q1, q2):
    """Apply CZ between qubits q1 and q2 (q1 < q2) on state of n qubits."""
    dim = 1 << n
    mask = np.arange(dim)
    both_one = ((mask >> q1) & 1) & ((mask >> q2) & 1) == 1
    new_state = state.copy()
    new_state[both_one] = -state[both_one]
    return new_state


def _circuit_state(n, layers, params):
    """Compute the state produced by the circuit.
    params: array of shape (layers, n)
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n

    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            state = _apply_ry(state, n, q, params[layer, q])
        # Apply CZ on adjacent pairs (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, n, q, q + 1)

    return state


def _hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling * sum Z_i Z_{i+1} - field * sum X_i"""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)

    # ZZ terms: diagonal
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            zval = (1 - 2 * bi) * (1 - 2 * bj)
            H[b, b] += -coupling * zval

    # X terms: off-diagonal
    for i in range(n):
        for b in range(dim):
            b_flip = b ^ (1 << i)
            H[b, b_flip] += -field

    return H


def _energy(state, H):
    """Compute <state|H|state>"""
    return np.real(np.vdot(state, H @ state))


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H"""
    H = _hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]


def _objective(params, n, coupling, field, layers):
    """Compute the objective: (circuit_energy - ground_energy) / n"""
    state = _circuit_state(n, layers, params)
    H = _hamiltonian(n, coupling, field)
    e_circuit = _energy(state, H)
    e_ground = _exact_ground_energy(n, coupling, field)
    return (e_circuit - e_ground) / n


def _nelder_mead(objective, x0, maxiter=200, tol=1e-8):
    """Nelder-Mead using scipy for robustness"""
    result = scipy.optimize.minimize(
        objective, x0, method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
    )
    return result.x, result.fun


def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin

    # Precompute ground energy and Hamiltonian
    e_ground = _exact_ground_energy(n, coupling, field)
    H = _hamiltonian(n, coupling, field)

    def obj(params):
        state = _circuit_state(n, layers, params)
        e_circuit = _energy(state, H)
        return (e_circuit - e_ground) / n

    # Initial guess: all zeros (product |0> state)
    x0 = np.zeros((layers, n))
    best_params = x0.copy()
    best_val = obj(x0)

    # Phase 1: Multi-restart with Nelder-Mead
    np.random.seed(42)
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            x_init = x0.copy()
        elif restart == 1:
            # Symmetric initial guess: all angles equal
            x_init = np.full((layers, n), np.pi / 4.0)
        else:
            x_init = np.random.uniform(-np.pi/2, np.pi/2, (layers, n))

        x_flat = x_init.flatten()

        def obj_flat(xf):
            return obj(xf.reshape((layers, n)))

        x_opt, f_opt = _nelder_mead(obj_flat, x_flat, maxiter=150, tol=1e-8)

        if f_opt < best_val:
            best_val = f_opt
            best_params = x_opt.reshape((layers, n))

        if time.time() > deadline:
            break

    # Phase 2: Fine-tune with L-BFGS-B from best point
    if time.time() < deadline - 5:
        x_best_flat = best_params.flatten()

        def obj_flat(xf):
            return obj(xf.reshape((layers, n)))

        try:
            result = scipy.optimize.minimize(
                obj_flat, x_best_flat, method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={'maxiter': 200, 'gtol': 1e-8}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.reshape((layers, n))
        except Exception:
            pass

    return best_params
# EVOLVE-BLOCK-END
