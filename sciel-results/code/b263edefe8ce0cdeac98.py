"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def _build_hamiltonian_terms(n, coupling, field):
    """
    Build list of terms for H = -coupling * sum_{i} Z_i Z_{i+1} - field * sum_i X_i
    Each term is (coeff, list_of_qubits, op_per_qubit)
    op_per_qubit is a dict mapping qubit index to 'Z' or 'X'
    """
    terms = []
    # ZZ terms
    for i in range(n - 1):
        terms.append((-coupling, [i, i + 1], {i: 'Z', i + 1: 'Z'}))
    # X terms
    for i in range(n):
        terms.append((-field, [i], {i: 'X'}))
    return terms

def _gate_matrix(op):
    """Return 2x2 matrix for single-qubit operation."""
    if op == 'X':
        return np.array([[0, 1], [1, 0]], dtype=complex)
    elif op == 'Z':
        return np.array([[1, 0], [0, -1]], dtype=complex)
    elif op == 'I':
        return np.array([[1, 0], [0, 1]], dtype=complex)
    else:
        raise ValueError(f"Unknown op: {op}")

def _ry_matrix(theta):
    """Ry(theta) = exp(-i theta Y / 2) = cos(theta/2) I - i sin(theta/2) Y"""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def _cz_matrix():
    """CZ gate matrix"""
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]], dtype=complex)

def _apply_single_qubit_gate(state, qubit, n, gate):
    """
    Apply a single-qubit gate to the state vector.
    state is a 2^n dimensional array.
    qubit is the qubit index (0 is least significant bit).
    """
    # Reshape state into tensor of shape (2,)*n
    shape = [2] * n
    state_tensor = state.reshape(shape)
    
    # Move the qubit axis to the front
    # We want to apply gate on axis `qubit`
    # Permute axes so that qubit is first
    perm = [qubit] + [i for i in range(n) if i != qubit]
    state_permuted = np.transpose(state_tensor, perm)
    
    # Apply gate: state_permuted has shape (2, 2, ..., 2)
    # gate has shape (2, 2)
    # New state = sum_{a,b} gate[a,b] * state_permuted[b, ...]
    # We can do this by reshaping to (2, -1), multiplying, and reshaping back
    orig_shape = state_permuted.shape
    state_reshaped = state_permuted.reshape(2, -1)
    new_state_reshaped = gate @ state_reshaped
    new_state_permuted = new_state_reshaped.reshape(orig_shape)
    
    # Move axes back
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    state_new = np.transpose(new_state_permuted, inv_perm)
    
    return state_new.reshape(-1)

def _apply_cz_gate(state, q1, q2, n):
    """
    Apply CZ gate between qubits q1 and q2.
    state is a 2^n dimensional array.
    """
    shape = [2] * n
    state_tensor = state.reshape(shape)
    
    # Move q1 and q2 to the front
    other_qubits = [i for i in range(n) if i != q1 and i != q2]
    perm = [q1, q2] + other_qubits
    state_permuted = np.transpose(state_tensor, perm)
    
    orig_shape = state_permuted.shape
    state_reshaped = state_permuted.reshape(4, -1)
    
    cz = _cz_matrix()
    new_state_reshaped = np.einsum('ab,b...->a...', cz, state_reshaped)
    
    new_state_permuted = new_state_reshaped.reshape(orig_shape)
    
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    state_new = np.transpose(new_state_permuted, inv_perm)
    
    return state_new.reshape(-1)

def _circuit_state(theta, n, layers):
    """
    Build the circuit state vector.
    theta is (layers, n) array.
    Start from |0>^n.
    Each layer: apply Ry(theta[layer, i]) to qubit i for all i.
    Between layers: apply CZ on adjacent pairs (0,1), (1,2), ..., (n-2, n-1).
    No entangler after final layer.
    """
    # Initial state |0>^n
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    
    for l in range(layers):
        # Apply Ry gates
        for i in range(n):
            gate = _ry_matrix(theta[l, i])
            state = _apply_single_qubit_gate(state, i, n, gate)
        
        # Apply CZ gates between layers (not after last layer)
        if l < layers - 1:
            for i in range(n - 1):
                state = _apply_cz_gate(state, i, i + 1, n)
    
    return state

def _expectation_value(state, terms, n):
    """
    Compute expectation value of Hamiltonian given state and terms.
    """
    energy = 0.0
    for coeff, qubits, ops in terms:
        # Compute <psi| O |psi>
        # O is a tensor product of single-qubit operators
        # We can apply the operator to the state and take inner product
        psi = state.copy()
        for q in qubits:
            op = _gate_matrix(ops[q])
            psi = _apply_single_qubit_gate(psi, q, n, op)
        energy += coeff * np.vdot(state, psi).real
    return energy

def _energy_and_grad(theta, n, coupling, field, layers):
    """
    Compute energy and gradient with respect to theta using finite differences
    for simplicity (could be improved with automatic differentiation or analytical gradients).
    For now, use central finite differences for gradient estimation.
    """
    energy = _expectation_value(_circuit_state(theta, n, layers), _build_hamiltonian_terms(n, coupling, field), n)
    
    # Gradient via finite differences
    eps = 1e-5
    grad = np.zeros_like(theta)
    for l in range(layers):
        for i in range(n):
            theta_plus = theta.copy()
            theta_plus[l, i] += eps
            e_plus = _expectation_value(_circuit_state(theta_plus, n, layers), _build_hamiltonian_terms(n, coupling, field), n)
            
            theta_minus = theta.copy()
            theta_minus[l, i] -= eps
            e_minus = _expectation_value(_circuit_state(theta_minus, n, layers), _build_hamiltonian_terms(n, coupling, field), n)
            
            grad[l, i] = (e_plus - e_minus) / (2 * eps)
    
    return energy, grad

def _exact_ground_energy(n, coupling, field):
    """
    Compute exact ground state energy by diagonalizing the Hamiltonian.
    For small n (4, 6), this is feasible.
    """
    # Build Hamiltonian matrix
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Helper to get index from bitstring
    def bitstring_to_index(bits):
        idx = 0
        for i, b in enumerate(bits):
            if b:
                idx |= (1 << i)
        return idx
    
    # ZZ terms
    for i in range(n - 1):
        for bits in itertools.product([0, 1], repeat=n):
            # Z_i Z_{i+1} = (+1) if bits[i] == bits[i+1], else (-1)
            z_val = 1 if bits[i] == bits[i + 1] else -1
            idx = bitstring_to_index(bits)
            H[idx, idx] += -coupling * z_val
    
    # X terms
    for i in range(n):
        for bits in itertools.product([0, 1], repeat=n):
            idx = bitstring_to_index(bits)
            # X flips bit i
            new_bits = list(bits)
            new_bits[i] = 1 - new_bits[i]
            new_idx = bitstring_to_index(new_bits)
            H[idx, new_idx] += -field
    
    # Diagonalize
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real

def angles(n, coupling, field, layers):
    """
    Optimize circuit parameters to minimize mean energy gap per spin.
    Returns theta of shape (layers, n).
    """
    # Compute exact ground energy
    e_ground = _exact_ground_energy(n, coupling, field)
    
    # Objective function: (circuit energy - ground energy) / n
    def objective(theta_flat):
        theta = theta_flat.reshape(layers, n)
        energy = _expectation_value(_circuit_state(theta, n, layers), _build_hamiltonian_terms(n, coupling, field), n)
        return (energy - e_ground) / n
    
    # Multiple initializations
    best_theta = None
    best_val = np.inf
    
    np.random.seed(42)
    
    # Try several random initializations plus some heuristic ones
    initializations = []
    
    # Heuristic: all zeros
    initializations.append(np.zeros((layers, n)))
    
    # Heuristic: pi/2 on last layer (like |+> state)
    theta_pi2 = np.zeros((layers, n))
    theta_pi2[-1] = np.pi / 2
    initializations.append(theta_pi2)
    
    # Random initializations
    for _ in range(10):
        initializations.append(np.random.uniform(-np.pi, np.pi, (layers, n)))
    
    # Run L-BFGS-B from each initialization
    for init in initializations:
        try:
            result = minimize(
                objective,
                init.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_theta = result.x.reshape(layers, n)
        except Exception:
            continue
    
    if best_theta is None:
        # Fallback to |+> state
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    return best_theta
# EVOLVE-BLOCK-END
