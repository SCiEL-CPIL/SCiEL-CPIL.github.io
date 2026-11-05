"""Optimize 3-layer Ry/CZ circuit for transverse-field Ising ground states."""
import numpy as np
from scipy.optimize import minimize
import time

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for open transverse-field Ising chain."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            zz_val = 1.0 if bit_i == bit_ip1 else -1.0
            H[state, state] += -coupling * zz_val
    
    # X_i terms
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[flipped, state] += -field
    
    return H

def _apply_ry(state, theta, qubit):
    """Apply Ry(theta) to qubit of state vector."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    dim = 1 << state.shape[0].bit_length() if hasattr(state.shape[0], 'bit_length') else len(state)
    n = int(np.log2(len(state)))
    for idx in range(len(state)):
        if (idx >> qubit) & 1 == 0:
            partner = idx | (1 << qubit)
            new_val = c * state[idx] - s * state[partner]
            state[idx] = new_val
    # Need to be more careful - let's use a different approach
    # Actually, let's build the Ry gate matrix and apply it
    return state

def _ry_gate(n, qubit, theta):
    """Build Ry gate as a 2^n x 2^n matrix."""
    dim = 1 << n
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    gate = np.zeros((dim, dim), dtype=complex)
    for idx in range(dim):
        if (idx >> qubit) & 1 == 0:
            partner = idx | (1 << qubit)
            gate[idx, idx] = c
            gate[idx, partner] = -s
            gate[partner, idx] = s
            gate[partner, partner] = c
        else:
            gate[idx, idx] = 1.0
    return gate

def _cz_gate(n, qubit1, qubit2):
    """Build CZ gate as a 2^n x 2^n matrix."""
    dim = 1 << n
    gate = np.eye(dim, dtype=complex)
    for idx in range(dim):
        if ((idx >> qubit1) & 1) and ((idx >> qubit2) & 1):
            gate[idx, idx] = -1.0
    return gate

def _circuit_state(n, params, layers):
    """Apply the circuit and return the final state vector."""
    state = np.zeros(1 << n, dtype=complex)
    state[0] = 1.0  # Start with |0>^n
    
    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            theta = params[layer * n + q]
            gate = _ry_gate(n, q, theta)
            state = gate @ state
        
        # Apply CZ between layers (not after final layer)
        if layer < layers - 1:
            for q in range(n - 1):
                gate = _cz_gate(n, q, q + 1)
                state = gate @ state
    
    return state

def _energy(state, H):
    """Compute expectation value of H in state."""
    return np.real(np.conj(state) @ H @ state)

def _objective(params, n, coupling, field, layers, H, ground_energy):
    """Objective: mean(circuit_energy - ground_energy)/n = (E_circuit - E_ground)/n"""
    state = _circuit_state(n, params, layers)
    e_circuit = _energy(state, H)
    return (e_circuit - ground_energy) / n

def angles(n, coupling, field, layers):
    """Optimize circuit parameters to approximate ground state."""
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Build Hamiltonian and find ground state
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    ground_energy = eigenvalues[0]
    
    n_params = layers * n
    
    # Try multiple initializations and keep the best
    best_params = None
    best_obj = np.inf
    
    # Initial guesses
    inits = [
        np.zeros(n_params),
        np.full(n_params, np.pi / 2),
        np.full(n_params, np.pi / 4),
        np.random.uniform(-np.pi, np.pi, n_params),
        np.random.uniform(-np.pi, np.pi, n_params),
    ]
    
    for init in inits:
        if time.time() - start_time > deadline - 10:
            break
        
        try:
            result = minimize(
                _objective,
                init,
                args=(n, coupling, field, layers, H, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            obj_val = result.fun
            if obj_val < best_obj:
                best_obj = obj_val
                best_params = result.x.copy()
        except Exception:
            continue
    
    # If no good result, try a few more random inits
    if best_params is None or best_obj > 0.5:
        for _ in range(10):
            if time.time() - start_time > deadline - 5:
                break
            init = np.random.uniform(-np.pi, np.pi, n_params)
            try:
                result = minimize(
                    _objective,
                    init,
                    args=(n, coupling, field, layers, H, ground_energy),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                obj_val = result.fun
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_params = result.x.copy()
            except Exception:
                continue
    
    if best_params is None:
        # Fallback: return zeros
        best_params = np.zeros(n_params)
    
    # Reshape to (layers, n)
    theta = best_params.reshape(layers, n)
    return theta
