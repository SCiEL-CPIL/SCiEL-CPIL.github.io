"""Optimized program: minimize circuit energy via scipy optimization on sparse H."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.optimize as opt
import time

def _build_hamiltonian(n, coupling, field):
    """Build H = -coupling*sum Z_i Z_{i+1} - field*sum X_i as dense matrix."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_j = (state >> (i + 1)) & 1
            zz = 1.0 if bit_i == bit_j else -1.0
            H[state, state] += -coupling * zz
    
    # X_i terms
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
            H[flipped, state] += -field
    
    return H


def _apply_ry(state, theta, qubit):
    """Apply Ry(theta) to qubit of state vector."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    # Ry = [[c, -s], [s, c]] in computational basis
    # For qubit: |0> -> c|0> + s|1>, |1> -> -s|0> + c|1>
    new_state = np.zeros_like(state)
    for idx in range(len(state)):
        bit = (idx >> qubit) & 1
        if bit == 0:
            # |0> component: c*|0> - s*|1>
            idx1 = idx | (1 << qubit)
            new_state[idx] += c * state[idx]
            new_state[idx1] += s * state[idx]
        else:
            # |1> component: -s*|0> + c*|1>
            idx0 = idx & ~(1 << qubit)
            new_state[idx] += c * state[idx]
            new_state[idx0] += -s * state[idx]
    return new_state


def _apply_cz(state, q1, q2):
    """Apply CZ between q1 and q2."""
    new_state = np.array(state, copy=True)
    for idx in range(len(state)):
        b1 = (idx >> q1) & 1
        b2 = (idx >> q2) & 1
        if b1 == 1 and b2 == 1:
            new_state[idx] = -state[idx]
    return new_state


def _circuit_state(theta, n, layers):
    """Apply the circuit: start |0>^n, each layer: Ry on all qubits, then CZ on adjacent pairs."""
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry(theta[layer, i]) to each qubit i
        for i in range(n):
            state = _apply_ry(state, theta[layer, i], i)
        # Apply CZ on adjacent pairs (except after final layer)
        if layer < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, i, i + 1)
    
    return state


def _energy(state, H):
    """Compute <psi|H|psi>."""
    return np.real(np.vdot(state, H @ state))


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    dim = 1 << n
    H = _build_hamiltonian(n, coupling, field)
    
    # Get exact ground state energy
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0]
    
    # Objective: minimize (E_circuit - e0) / n
    def objective(theta_flat):
        theta = theta_flat.reshape(layers, n)
        state = _circuit_state(theta, n, layers)
        e = _energy(state, H)
        return (e - e0) / n
    
    # Initial guess: all zeros
    x0 = np.zeros(layers * n)
    
    # Try to optimize
    best_theta = x0.copy()
    best_val = objective(x0)
    
    # Use L-BFGS-B with finite differences
    try:
        res = opt.minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        if res.fun < best_val:
            best_theta = res.x
            best_val = res.fun
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        return best_theta.reshape(layers, n)
    
    # Try Nelder-Mead as backup/improvement
    try:
        res2 = opt.minimize(
            objective,
            best_theta,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if res2.fun < best_val:
            best_theta = res2.x
            best_val = res2.fun
    except Exception:
        pass
    
    return best_theta.reshape(layers, n)
