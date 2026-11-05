"""Optimized quantum circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 150.0  # wall-clock kill limit
_START_TIME = time.time()


def _check_deadline(margin=2.0):
    return (time.time() - _START_TIME) < (_DEADLINE - margin)


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dims = [2] * n
    total_dim = 2 ** n
    H = np.zeros((total_dim, total_dim), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        Z_i = np.zeros((total_dim, total_dim), dtype=complex)
        Z_ip1 = np.zeros((total_dim, total_dim), dtype=complex)
        for state in range(total_dim):
            z_i = 1.0 if (state >> i) & 1 == 0 else -1.0
            z_ip1 = 1.0 if (state >> (i + 1)) & 1 == 0 else -1.0
            Z_i[state, state] = z_i
            Z_ip1[state, state] = z_ip1
        H += -coupling * (Z_i @ Z_ip1)
    
    # X terms: -field * X_i
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    for i in range(n):
        X_i = np.zeros((total_dim, total_dim), dtype=complex)
        for state in range(total_dim):
            flipped = state ^ (1 << i)
            X_i[flipped, state] = 1.0
        H += -field * X_i
    
    return H


def _get_ground_energy(H):
    """Get the exact ground state energy via eigendecomposition."""
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real


def _apply_ry(state, theta, qubit):
    """Apply Ry(theta) to qubit in state vector."""
    n = int(np.log2(len(state)))
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    
    new_state = np.zeros_like(state)
    for state_idx in range(len(state)):
        # Get bit value for this qubit
        bit = (state_idx >> qubit) & 1
        # Ry = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
        if bit == 0:
            # |0> -> c|0> + s|1>
            new_state[state_idx] += c * state[state_idx]
            flipped = state_idx | (1 << qubit)
            new_state[flipped] += s * state[state_idx]
        else:
            # |1> -> -s|0> + c|1>
            new_state[state_idx] += c * state[state_idx]
            flipped = state_idx ^ (1 << qubit)
            new_state[flipped] += -s * state[state_idx]
    return new_state


def _apply_cz(state, q1, q2):
    """Apply CZ gate between q1 and q2."""
    new_state = np.zeros_like(state)
    for state_idx in range(len(state)):
        b1 = (state_idx >> q1) & 1
        b2 = (state_idx >> q2) & 1
        phase = -1.0 if (b1 == 1 and b2 == 1) else 1.0
        new_state[state_idx] = phase * state[state_idx]
    return new_state


def _circuit_state(params, n, layers):
    """Build the circuit state from parameters."""
    total_dim = 2 ** n
    state = np.zeros(total_dim, dtype=complex)
    state[0] = 1.0  # Start with |0>^n
    
    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            state = _apply_ry(state, params[layer, q], q)
        
        # Apply CZ between adjacent pairs (except after final layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1)
    
    return state


def _circuit_energy(params, n, coupling, field, layers, H):
    """Compute the energy of the circuit state."""
    state = _circuit_state(params, n, layers)
    energy = np.vdot(state, H @ state).real
    return energy


def _objective(params_flat, n, coupling, field, layers, H, ground_energy):
    """Objective function: mean energy gap per spin."""
    params = params_flat.reshape(layers, n)
    energy = _circuit_energy(params, n, coupling, field, layers, H)
    return (energy - ground_energy) / n


def _objective_multi(params_flat, configs, Hs, ground_energies):
    """Multi-field objective: sum of mean energy gaps."""
    total = 0.0
    for i, (n, coupling, field, layers) in enumerate(configs):
        params = params_flat[i * layers * n:(i + 1) * layers * n].reshape(layers, n)
        energy = _circuit_energy(params, n, coupling, field, layers, Hs[i])
        total += (energy - ground_energies[i]) / n
    return total


def angles(n, coupling, field, layers):
    """Optimize circuit parameters for the given Ising chain."""
    global _START_TIME
    _START_TIME = time.time()
    
    total_dim = 2 ** n
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _get_ground_energy(H)
    
    # Initial guess: all zeros
    best_params = np.zeros((layers, n))
    best_score = np.inf
    
    # Try multiple initial points
    np.random.seed(42)
    n_trials = 5
    for trial in range(n_trials):
        if not _check_deadline():
            break
        
        if trial == 0:
            x0 = np.zeros(layers * n)
        else:
            x0 = np.random.uniform(-np.pi, np.pi, size=layers * n)
        
        try:
            result = minimize(
                _objective,
                x0,
                args=(n, coupling, field, layers, H, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 50, 'maxfun': 200, 'ftol': 1e-8}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.reshape(layers, n)
                
        except Exception:
            continue
    
    return best_params
# EVOLVE-BLOCK-END
