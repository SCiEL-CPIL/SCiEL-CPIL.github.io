"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _ry_matrix(theta):
    """Ry(theta) = exp(-i theta Y / 2)"""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def _cz_matrix():
    """CZ gate matrix (4x4)"""
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]], dtype=complex)

def _x_matrix():
    return np.array([[0, 1], [1, 0]], dtype=complex)

def _z_matrix():
    return np.array([[1, 0], [0, -1]], dtype=complex)

def _i_matrix():
    return np.eye(2, dtype=complex)

def _build_hamiltonian_terms(n, coupling, field):
    """
    Build list of terms for H = -coupling * sum_{i} Z_i Z_{i+1} - field * sum_i X_i
    Each term is (coeff, list_of_qubits, op_per_qubit)
    """
    terms = []
    for i in range(n - 1):
        terms.append((-coupling, [i, i + 1], {i: 'Z', i + 1: 'Z'}))
    for i in range(n):
        terms.append((-field, [i], {i: 'X'}))
    return terms

def _get_op_matrix(op):
    if op == 'X':
        return _x_matrix()
    elif op == 'Z':
        return _z_matrix()
    elif op == 'I':
        return _i_matrix()
    else:
        raise ValueError(f"Unknown op: {op}")

def _apply_gate_to_tensor(tensor, qubit, gate, n):
    """
    Apply a 2x2 gate to qubit `qubit` of a tensor with shape (2,)*n.
    Returns new tensor of same shape.
    """
    # Permute so qubit is first axis
    perm = [qubit] + [i for i in range(n) if i != qubit]
    t_perm = np.transpose(tensor, perm)
    orig_shape = t_perm.shape
    
    # Reshape to (2, -1)
    t_reshaped = t_perm.reshape(2, -1)
    new_reshaped = gate @ t_reshaped
    new_perm = new_reshaped.reshape(orig_shape)
    
    # Inverse permutation
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return np.transpose(new_perm, inv_perm)

def _apply_cz_to_tensor(tensor, q1, q2, n):
    """
    Apply CZ gate between qubits q1 and q2 of a tensor with shape (2,)*n.
    """
    cz = _cz_matrix()
    other = [i for i in range(n) if i != q1 and i != q2]
    perm = [q1, q2] + other
    t_perm = np.transpose(tensor, perm)
    orig_shape = t_perm.shape
    
    t_reshaped = t_perm.reshape(4, -1)
    new_reshaped = np.einsum('ab,b...->a...', cz, t_reshaped)
    new_perm = new_reshaped.reshape(orig_shape)
    
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return np.transpose(new_perm, inv_perm)

def _circuit_tensor(theta, n, layers):
    """
    Build the circuit state as a tensor of shape (2,)*n.
    Start from |0>^n.
    Each layer: apply Ry(theta[layer, i]) to qubit i for all i.
    Between layers: apply CZ on adjacent pairs (0,1), (1,2), ..., (n-2, n-1).
    No entangler after final layer.
    """
    # Initial state |0>^n: tensor with all indices 0
    state = np.zeros((2,) * n, dtype=complex)
    state[(0,) * n] = 1.0
    
    for l in range(layers):
        # Apply Ry gates
        for i in range(n):
            gate = _ry_matrix(theta[l, i])
            state = _apply_gate_to_tensor(state, i, gate, n)
        
        # Apply CZ gates between layers (not after last layer)
        if l < layers - 1:
            for i in range(n - 1):
                state = _apply_cz_to_tensor(state, i, i + 1, n)
    
    return state

def _expectation_from_tensor(state, terms, n):
    """
    Compute expectation value of Hamiltonian given state tensor and terms.
    """
    energy = 0.0
    for coeff, qubits, ops in terms:
        psi = state.copy()
        for q in qubits:
            op = _get_op_matrix(ops[q])
            psi = _apply_gate_to_tensor(psi, q, op, n)
        energy += coeff * np.vdot(state, psi).real
    return energy

def _energy_and_grad(theta, n, coupling, field, layers):
    """
    Compute energy and gradient with respect to theta using finite differences
    on the tensor-based state.
    """
    state = _circuit_tensor(theta, n, layers)
    terms = _build_hamiltonian_terms(n, coupling, field)
    energy = _expectation_from_tensor(state, terms, n)
    
    # Gradient via finite differences
    eps = 1e-6
    grad = np.zeros_like(theta)
    for l in range(layers):
        for i in range(n):
            theta_plus = theta.copy()
            theta_plus[l, i] += eps
            state_plus = _circuit_tensor(theta_plus, n, layers)
            e_plus = _expectation_from_tensor(state_plus, terms, n)
            
            theta_minus = theta.copy()
            theta_minus[l, i] -= eps
            state_minus = _circuit_tensor(theta_minus, n, layers)
            e_minus = _expectation_from_tensor(state_minus, terms, n)
            
            grad[l, i] = (e_plus - e_minus) / (2 * eps)
    
    return energy, grad

def _exact_ground_energy(n, coupling, field):
    """
    Compute exact ground state energy by diagonalizing the Hamiltonian.
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    def bitstring_to_index(bits):
        idx = 0
        for i, b in enumerate(bits):
            if b:
                idx |= (1 << i)
        return idx
    
    # ZZ terms
    for i in range(n - 1):
        for bits in np.ndindex(*([2] * n)):
            z_val = 1 if bits[i] == bits[i + 1] else -1
            idx = bitstring_to_index(bits)
            H[idx, idx] += -coupling * z_val
    
    # X terms
    for i in range(n):
        for bits in np.ndindex(*([2] * n)):
            idx = bitstring_to_index(bits)
            new_bits = list(bits)
            new_bits[i] = 1 - new_bits[i]
            new_idx = bitstring_to_index(new_bits)
            H[idx, new_idx] += -field
    
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real

def angles(n, coupling, field, layers):
    """
    Optimize circuit parameters to minimize mean energy gap per spin.
    Returns theta of shape (layers, n).
    """
    e_ground = _exact_ground_energy(n, coupling, field)
    
    def objective(theta_flat):
        theta = theta_flat.reshape(layers, n)
        energy = _expectation_from_tensor(_circuit_tensor(theta, n, layers), _build_hamiltonian_terms(n, coupling, field), n)
        return (energy - e_ground) / n
    
    def objective_grad(theta_flat):
        theta = theta_flat.reshape(layers, n)
        energy, grad = _energy_and_grad(theta, n, coupling, field, layers)
        return (energy - e_ground) / n, (grad / n).flatten()
    
    # Multiple initializations
    best_theta = None
    best_val = np.inf
    
    np.random.seed(42)
    
    initializations = []
    initializations.append(np.zeros((layers, n)))
    
    theta_pi2 = np.zeros((layers, n))
    theta_pi2[-1] = np.pi / 2
    initializations.append(theta_pi2)
    
    theta_all_pi2 = np.full((layers, n), np.pi / 2)
    initializations.append(theta_all_pi2)
    
    theta_small = np.full((layers, n), 0.1)
    initializations.append(theta_small)
    
    for _ in range(8):
        initializations.append(np.random.uniform(-np.pi, np.pi, (layers, n)))
    
    for init in initializations:
        try:
            result = minimize(
                objective,
                init.flatten(),
                jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_theta = result.x.reshape(layers, n)
        except Exception:
            continue
    
    if best_theta is None:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    return best_theta
# EVOLVE-BLOCK-END
