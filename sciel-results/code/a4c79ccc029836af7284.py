"""Quantum information: shallow-circuit Ising ground states
Uses sparse Hamiltonian and direct energy minimization via scipy optimizer.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import lil_matrix, diags, kron
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
import itertools


def _build_hamiltonian(n, coupling, field):
    """Build sparse Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i for open chain."""
    # Single-qubit operators
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Build full operators using kron
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i + 1] = Z
        H -= coupling * kron_all(ops)
    
    # X terms: -field * X_i
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= field * kron_all(ops)
    
    return H


def _apply_circuit(state, theta):
    """Apply the 3-layer Ry/CZ circuit to initial state |0>^n.
    theta has shape (layers, n). Returns final state vector.
    """
    n = theta.shape[1]
    layers = theta.shape[0]
    
    # Single-qubit operators
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def ry(theta_val):
        c = np.cos(theta_val / 2)
        s = np.sin(theta_val / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    
    def cz():
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, -1]], dtype=complex)
    
    def apply_single_qubit(state, op, qubit, n_qubits):
        """Apply single-qubit op to given qubit in multi-qubit state."""
        # Reshape state to tensor
        dim = 2 ** n_qubits
        state_tensor = state.reshape([2] * n_qubits)
        
        # Apply op on the specified qubit axis
        # op is 2x2, apply to axis 'qubit'
        # Use einsum or manual
        new_tensor = np.zeros_like(state_tensor)
        for i in range(2):
            for j in range(2):
                # op[i,j] * state_tensor[..., j, ...] -> new_tensor[..., i, ...]
                # We need to be careful with axis ordering
                # Let's use a simpler approach: iterate over all indices
                pass
        
        # Simpler: use kron decomposition
        # Build full operator
        ops = [I] * n_qubits
        ops[qubit] = op
        full_op = ops[0]
        for k in range(1, n_qubits):
            full_op = np.kron(full_op, ops[k])
        return full_op @ state
    
    def apply_cz_pair(state, q1, q2, n_qubits):
        """Apply CZ between qubits q1 and q2."""
        # Build CZ on the two qubits, identity elsewhere
        # CZ = |00><00| + |01><01| + |10><10| - |11><11|
        # For simplicity, build full operator
        ops = [I] * n_qubits
        # Need to place CZ on qubits q1 and q2
        # This requires careful kron ordering
        # Let's build it by iterating
        dim = 2 ** n_qubits
        H_cz = np.zeros((dim, dim), dtype=complex)
        
        # CZ acts as: if both qubits are 1, flip sign
        for basis_idx in range(dim):
            bits = [(basis_idx >> k) & 1 for k in range(n_qubits)]
            if bits[q1] == 1 and bits[q2] == 1:
                H_cz[basis_idx, basis_idx] = -1
            else:
                H_cz[basis_idx, basis_idx] = 1
        
        return H_cz @ state
    
    # Start from |0>^n
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry(theta[layer, i]) to each qubit i
        for i in range(n):
            op = ry(theta[layer, i])
            state = apply_single_qubit(state, op, i, n)
        
        # Apply CZ on adjacent pairs in ascending order, except after final layer
        if layer < layers - 1:
            for i in range(n - 1):
                state = apply_cz_pair(state, i, i + 1, n)
    
    return state


def _energy(state, H):
    """Compute expectation value <state|H|state>."""
    return np.real(np.vdot(state, H @ state))


def _objective(params, n, coupling, field, H):
    """Objective function: mean energy per spin."""
    theta = params.reshape(3, n)
    state = _apply_circuit(theta, n)
    e = _energy(state, H)
    return e / n


def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy per spin.
    
    Returns array of shape (layers, n) with optimized angles.
    """
    # Build Hamiltonian
    H = _build_hamiltonian(n, coupling, field)
    
    # Initial guess: try product |+> state (theta[-1] = pi/2, others 0)
    theta0 = np.zeros((layers, n))
    theta0[-1] = np.pi / 2
    
    # Try a few different initial guesses and pick the best
    best_theta = theta0.copy()
    best_energy = np.inf
    
    # Initial guess 1: product |+>
    candidates = [theta0.copy()]
    
    # Initial guess 2: all zeros
    candidates.append(np.zeros((layers, n)))
    
    # Initial guess 3: small random perturbations
    rng = np.random.RandomState(42)
    for _ in range(5):
        cand = rng.uniform(-np.pi/4, np.pi/4, (layers, n))
        candidates.append(cand)
    
    # Initial guess 4: try to match mean field solution
    # For transverse field Ising, approximate angles
    for trial in range(3):
        cand = np.zeros((layers, n))
        # Last layer: try different values
        cand[-1] = np.linspace(0.2, 1.2, n)
        candidates.append(cand)
    
    for theta_init in candidates:
        params_init = theta_init.flatten()
        
        # Use L-BFGS-B optimizer
        try:
            result = minimize(
                _objective,
                params_init,
                args=(n, coupling, field, H),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            e = _objective(result.x, n, coupling, field, H)
            if e < best_energy:
                best_energy = e
                best_theta = result.x.reshape(layers, n)
        except Exception:
            continue
    
    # Clamp angles to reasonable range
    best_theta = np.clip(best_theta, -np.pi, np.pi)
    
    return best_theta
# EVOLVE-BLOCK-END
