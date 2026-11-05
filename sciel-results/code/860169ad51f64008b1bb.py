"""Ising ground state approximation with 3-layer Ry/CZ circuit.

Strategy:
  1. Build exact ground state via dense eigensolver.
  2. Use multiple physically-motivated initial guesses (symmetry, product states).
  3. Refine with L-BFGS-B on the mean energy gap objective.
  4. Return best parameters found within the time budget.
"""
import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = None


def _set_deadline(start_time, margin=130.0):
    global _DEADLINE
    _DEADLINE = start_time + margin


def _time_left():
    if _DEADLINE is None:
        return 120.0
    return _DEADLINE - time.time()


def _exact_ground_state(n, coupling, field):
    """Compute the exact ground state of the transverse-field Ising chain."""
    dim = 2 ** n
    # Build H = -coupling * sum_i Z_i Z_{i+1} - field * sum_i X_i
    I = np.eye(2)
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])

    H = np.zeros((dim, dim), dtype=complex)

    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        term = np.eye(2)
        for j in range(n):
            if j == i:
                term = np.kron(term, Z)
            elif j == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H -= coupling * term

    # X_i terms
    for i in range(n):
        term = np.eye(2)
        for j in range(n):
            if j == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H -= field * term

    # Eigendecomposition
    vals, vecs = np.linalg.eigh(H)
    ground_energy = vals[0].real
    ground_state = vecs[:, 0]
    return ground_state, ground_energy


def _circuit_state(params, n, layers):
    """Apply the 3-layer Ry/CZ circuit to |0>^n and return the state vector."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0>^n

    I = np.eye(2)
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])

    for layer in range(layers):
        # Apply Ry(theta) to every qubit
        thetas = params[layer]
        for q in range(n):
            theta = thetas[q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            Ry = np.array([[c, -s], [s, c]])

            # Build full gate
            gate = np.eye(2)
            for j in range(n):
                if j == q:
                    gate = np.kron(gate, Ry)
                else:
                    gate = np.kron(gate, I)
            state = gate @ state

        # Apply CZ between adjacent pairs (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                CZ = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, -1]], dtype=complex)
                # Build full CZ gate
                gate = np.eye(2)
                for j in range(n):
                    if j == q or j == q + 1:
                        # We'll handle these two together
                        pass
                    else:
                        gate = np.kron(gate, I)
                # Actually build it properly
                gate = np.ones((dim, dim), dtype=complex)
                # For each basis state, if both qubits q and q+1 are 1, flip sign
                for idx in range(dim):
                    bits = [(idx >> k) & 1 for k in range(n)]
                    if bits[q] == 1 and bits[q + 1] == 1:
                        gate[idx, idx] = -1.0
                state = gate @ state

    return state


def _energy(state, n, coupling, field):
    """Compute <state|H|state> for the Ising Hamiltonian."""
    dim = 2 ** n
    I = np.eye(2)
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])

    energy = 0.0

    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        term = np.eye(2)
        for j in range(n):
            if j == i or j == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        energy -= coupling * np.vdot(state, term @ state).real

    # X_i terms
    for i in range(n):
        term = np.eye(2)
        for j in range(n):
            if j == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        energy -= field * np.vdot(state, term @ state).real

    return energy


def _objective(params_flat, n, coupling, field, layers, ground_energy):
    """Objective: mean energy gap = (circuit_energy - ground_energy) / n."""
    params = params_flat.reshape(layers, n)
    state = _circuit_state(params, n, layers)
    e = _energy(state, n, coupling, field)
    return (e - ground_energy) / n


def angles(n, coupling, field, layers):
    """Return optimal circuit parameters (layers, n) for the Ising chain."""
    start_time = time.time()
    _set_deadline(start_time, margin=135.0)

    # Compute exact ground state and energy
    ground_state, ground_energy = _exact_ground_state(n, coupling, field)

    best_params = np.zeros((layers, n))
    best_obj = np.inf

    # Generate initial guesses
    guesses = []

    # 1. All zeros (|0>^n state)
    guesses.append(np.zeros((layers, n)))

    # 2. Last layer pi/2 (|+> state, the seed)
    g = np.zeros((layers, n))
    g[-1] = np.pi / 2
    guesses.append(g.copy())

    # 3. Uniform small angles
    for val in [0.1, 0.2, 0.3, 0.5]:
        g = np.full((layers, n), val)
        guesses.append(g)

    # 4. Symmetry-based: alternate angles for antiferromagnetic-like patterns
    for base in [0.2, 0.4, 0.6]:
        g = np.zeros((layers, n))
        for l in range(layers):
            for q in range(n):
                g[l, q] = base if q % 2 == 0 else -base
        guesses.append(g)

    # 5. Gradient-based from |+> with varying last layer
    for last_theta in [0.3, 0.5, 0.7, np.pi / 3]:
        g = np.zeros((layers, n))
        g[-1] = last_theta
        guesses.append(g)

    # 6. Random guesses with fixed seed for reproducibility
    rng = np.random.default_rng(seed=42 + n * 7 + int(round(field * 1000)))
    for _ in range(6):
        g = rng.uniform(-0.8, 0.8, size=(layers, n))
        guesses.append(g)

    # Try each guess with L-BFGS-B refinement
    bounds = [(-np.pi, np.pi)] * (layers * n)

    for idx, guess in enumerate(guesses):
        if _time_left() < 5.0:
            break

        x0 = guess.flatten()
        try:
            result = minimize(
                _objective, x0,
                args=(n, coupling, field, layers, ground_energy),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.reshape(layers, n)
        except Exception:
            pass

    # Second stage: finer refinement from best so far
    if _time_left() > 10.0:
        x0 = best_params.flatten()
        try:
            result = minimize(
                _objective, x0,
                args=(n, coupling, field, layers, ground_energy),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.reshape(layers, n)
        except Exception:
            pass

    # If still time, try a few more random restarts from perturbed best
    if _time_left() > 15.0:
        rng2 = np.random.default_rng(seed=99 + n + int(round(field * 1000)))
        for _ in range(3):
            if _time_left() < 8.0:
                break
            perturbation = rng2.normal(0, 0.1, size=(layers, n))
            x0 = best_params.flatten() + perturbation.flatten()
            # Clip to bounds
            x0 = np.clip(x0, -np.pi, np.pi)
            try:
                result = minimize(
                    _objective, x0,
                    args=(n, coupling, field, layers, ground_energy),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-11, 'gtol': 1e-9}
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x.reshape(layers, n)
            except Exception:
                pass

    return best_params
