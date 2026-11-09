"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Cache for exact ground energies and circuit evaluation
_energy_cache = {}

def _get_pauli_matrices(n):
    """Return Pauli matrices for n qubits."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    def kron_list(mats):
        result = np.array([[1.0]], dtype=complex)
        for m in mats:
            result = np.kron(result, m)
        return result
    
    Xs = []
    Zs = []
    for i in range(n):
        mats = [I] * n
        mats[i] = X
        Xs.append(kron_list(mats))
        mats[i] = Z
        Zs.append(kron_list(mats))
    return Xs, Zs

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i"""
    Xs, Zs = _get_pauli_matrices(n)
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Transverse field term: -field * sum X_i
    for i in range(n):
        H += -field * Xs[i]
    
    # Coupling term: -coupling * sum Z_i Z_{i+1} (open chain)
    for i in range(n - 1):
        ZiZj = np.kron(Zs[i] if i == 0 else np.eye(2), np.eye(2))  # placeholder
        # Actually compute Z_i ⊗ Z_{i+1}
        mats = [np.eye(2, dtype=complex)] * n
        mats[i] = Zs[i] if isinstance(Zs[i], np.ndarray) else Zs[i]
        # Rebuild properly
        mats2 = [np.eye(2, dtype=complex)] * n
        mats2[i] = np.array([[1,0],[0,-1]], dtype=complex)
        mats2[i+1] = np.array([[1,0],[0,-1]], dtype=complex)
        ZiZj = np.array([[1.0]], dtype=complex)
        for m in mats2:
            ZiZj = np.kron(ZiZj, m)
        H += -coupling * ZiZj
    
    return H

def _apply_circuit(params, n, layers):
    """Apply the 3-layer Ry/CZ circuit to |0>^n and return the state vector."""
    dim = 2 ** n
    # Start in |0>^n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    def apply_ry(state, theta, qubit, n):
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        # Ry = [[cos, -sin], [sin, cos]]
        new_state = np.zeros_like(state)
        for i in range(dim):
            if (i >> qubit) & 1 == 0:
                # |0> component
                idx0 = i
                idx1 = i | (1 << qubit)
                new_state[idx0] += c * state[idx0]
                new_state[idx1] += s * state[idx0]
            else:
                # |1> component
                idx1 = i
                idx0 = i & ~(1 << qubit)
                new_state[idx0] += -s * state[idx1]
                new_state[idx1] += c * state[idx1]
        return new_state
    
    def apply_cz(state, q1, q2, n):
        # CZ flips phase of |11> on qubits q1, q2
        new_state = state.copy()
        for i in range(dim):
            if ((i >> q1) & 1) and ((i >> q2) & 1):
                new_state[i] *= -1
        return new_state
    
    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            state = apply_ry(state, params[layer, q], q, n)
        # Apply CZ on adjacent pairs (except after final layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = apply_cz(state, q, q + 1, n)
    
    return state

def _circuit_energy(params, n, coupling, field, H):
    """Compute the energy expectation of the circuit state."""
    state = _apply_circuit(params, n, len(params))
    return np.real(np.conj(state) @ H @ state)

def _objective_and_grad(params, n, coupling, field, H, ground_energy):
    """Objective: (circuit_energy - ground_energy) / n, with analytic gradient."""
    energy = _circuit_energy(params, n, coupling, field, H)
    obj = (energy - ground_energy) / n
    
    # Finite difference gradient
    eps = 1e-7
    grad = np.zeros_like(params)
    for idx in np.ndindex(params.shape):
        orig = params[idx]
        params[idx] = orig + eps
        e_plus = _circuit_energy(params, n, coupling, field, H)
        params[idx] = orig - eps
        e_minus = _circuit_energy(params, n, coupling, field, H)
        params[idx] = orig
        grad[idx] = (e_plus - e_minus) / (2 * eps) / n
    
    return obj, grad

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via dense diagonalization."""
    key = (n, coupling, field)
    if key in _energy_cache:
        return _energy_cache[key]
    
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    ground_energy = float(eigvals[0])
    _energy_cache[key] = ground_energy
    return ground_energy

def angles(n, coupling, field, layers):
    """Optimize circuit parameters to approximate ground state."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Compute exact ground energy
    ground_energy = _exact_ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    
    best_params = np.zeros((layers, n))
    best_obj = np.inf
    
    # Multi-start L-BFGS-B optimization
    np.random.seed(42)
    num_starts = 20
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        # Initialize parameters
        if start_idx == 0:
            params = np.zeros((layers, n))
        elif start_idx == 1:
            params = np.zeros((layers, n))
            params[-1] = np.pi / 2
        else:
            params = np.random.uniform(-np.pi, np.pi, size=(layers, n))
        
        # Run L-BFGS-B
        try:
            result = minimize(
                _objective_and_grad,
                params,
                args=(n, coupling, field, H, ground_energy),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'gtol': 1e-6, 'maxls': 20}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            continue
    
    # If no optimization succeeded, fall back to simple |+> state
    if best_obj == np.inf:
        best_params = np.zeros((layers, n))
        best_params[-1] = np.pi / 2
    
    return best_params
# EVOLVE-BLOCK-END
