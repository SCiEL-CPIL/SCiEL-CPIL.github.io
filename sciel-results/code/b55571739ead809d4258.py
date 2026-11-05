import numpy as np
from scipy.optimize import minimize
import time

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for the transverse-field Ising chain."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        for state in range(dim):
            zi = 1 if (state >> i) & 1 == 0 else -1
            zj = 1 if (state >> (i + 1)) & 1 == 0 else -1
            H[state, state] += -coupling * zi * zj
    
    # X terms: -field * sum X_i
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[flipped, state] += -field
    
    return H

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real

def _apply_ry(state, theta, qubit, n):
    """Apply Ry(theta) to qubit in state vector."""
    dim = 1 << n
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    new_state = np.zeros_like(state)
    for state_idx in range(dim):
        if (state_idx >> qubit) & 1 == 0:
            # |0> component
            idx1 = state_idx
            idx0 = state_idx
            # Ry maps |0> -> c|0> - i*s|1>
            new_state[idx0] += c * state[idx0]
            new_state[idx1 ^ (1 << qubit)] += -1j * s * state[idx0]
        else:
            # |1> component
            idx1 = state_idx
            # Ry maps |1> -> -i*s|0> + c|1>
            new_state[idx1 ^ (1 << qubit)] += -1j * s * state[idx1]
            new_state[idx1] += c * state[idx1]
    return new_state

def _apply_cz(state, q1, q2, n):
    """Apply CZ to qubits q1, q2."""
    dim = 1 << n
    new_state = np.zeros_like(state)
    for state_idx in range(dim):
        b1 = (state_idx >> q1) & 1
        b2 = (state_idx >> q2) & 1
        phase = -1 if (b1 == 1 and b2 == 1) else 1
        new_state[state_idx] = phase * state[state_idx]
    return new_state

def _circuit_state(theta, n, layers):
    """Build the circuit state vector from angles."""
    dim = 1 << n
    # Start in |0>^n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            state = _apply_ry(state, theta[layer, q], q, n)
        # Apply CZ between adjacent pairs (not after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1, n)
    
    return state

def _energy_expectation(state, n, coupling, field):
    """Compute <state|H|state>."""
    dim = 1 << n
    E = 0.0
    
    # ZZ terms
    for i in range(n - 1):
        for state_idx in range(dim):
            zi = 1 if (state_idx >> i) & 1 == 0 else -1
            zj = 1 if (state_idx >> (i + 1)) & 1 == 0 else -1
            E += -coupling * zi * zj * (abs(state[state_idx]) ** 2)
    
    # X terms
    for i in range(n):
        for state_idx in range(dim):
            flipped = state_idx ^ (1 << i)
            E += -field * np.real(state[flipped].conj() * state[state_idx])
    
    return E

def _objective(params, n, coupling, field, layers, E0):
    """Objective: mean energy gap per spin."""
    theta = params.reshape(layers, n)
    state = _circuit_state(theta, n, layers)
    E = _energy_expectation(state, n, coupling, field)
    return (E - E0) / n

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Compute exact ground energy
    E0 = _exact_ground_energy(n, coupling, field)
    
    # Initial guess: try a few different starting points
    best_params = None
    best_val = np.inf
    
    # Try several initializations
    inits = []
    # Zero initialization
    inits.append(np.zeros((layers, n)))
    # Pi/2 on last layer (product |+>)
    t = np.zeros((layers, n))
    t[-1] = np.pi / 2
    inits.append(t)
    # Small random perturbations
    for _ in range(4):
        t = np.random.uniform(-0.5, 0.5, (layers, n))
        inits.append(t)
    
    for init in inits:
        if time.time() > deadline - 10:
            break
        
        flat_init = init.flatten()
        
        try:
            result = minimize(
                _objective,
                flat_init,
                args=(n, coupling, field, layers, E0),
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    if best_params is None:
        # Fallback
        return np.zeros((layers, n))
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _objective,
                best_params,
                args=(n, coupling, field, layers, E0),
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < best_val:
                best_params = result.x.copy()
        except Exception:
            pass
    
    theta = best_params.reshape(layers, n)
    # Wrap to [-pi, pi]
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    return theta
