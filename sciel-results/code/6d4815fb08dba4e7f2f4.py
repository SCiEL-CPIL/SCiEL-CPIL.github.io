# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # Pauli X
    X = np.array([[0, 1], [1, 0]], dtype=np.float64)
    Z = np.array([[1, 0], [0, -1]], dtype=np.float64)
    I = np.eye(2, dtype=np.float64)
    
    def kron_list(ops):
        result = np.array([[1.0]])
        for op in ops:
            result = np.kron(result, op)
        return result
    
    # ZZ terms: -coupling * sum_{i=0}^{n-2} Z_i Z_{i+1}
    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i + 1] = Z
        H -= coupling * kron_list(ops)
    
    # X terms: -field * sum_{i=0}^{n-1} X_i
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= field * kron_list(ops)
    
    return H


def _apply_ry(state, theta, qubit, n):
    """Apply Ry(theta) to qubit of an n-qubit state."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    
    # Ry = [[c, -s], [s, c]]
    dim = 2 ** n
    new_state = np.zeros_like(state)
    
    for idx in range(dim):
        bit = (idx >> qubit) & 1
        if bit == 0:
            idx1 = idx | (1 << qubit)
            new_state[idx] = c * state[idx] - s * state[idx1]
            new_state[idx1] = s * state[idx] + c * state[idx1]
        # else: already handled when bit==0 for the pair
    
    return new_state


def _apply_cz(state, q1, q2, n):
    """Apply CZ between qubits q1 and q2."""
    dim = 2 ** n
    new_state = np.array(state, dtype=np.complex128)
    
    for idx in range(dim):
        b1 = (idx >> q1) & 1
        b2 = (idx >> q2) & 1
        if b1 == 1 and b2 == 1:
            new_state[idx] = -state[idx]
    
    return new_state


def _circuit_state(params, n, layers):
    """Prepare the circuit state from parameters.
    params: flat array of shape (layers * n)
    Layer order: for each layer, apply Ry to all qubits, then CZ on adjacent pairs (except after last layer).
    """
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            theta = params[layer * n + q]
            state = _apply_ry(state, theta, q, n)
        
        # Apply CZ on adjacent pairs (not after final layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1, n)
    
    return state


def _energy(state, H):
    """Compute <state|H|state> for a real or complex state."""
    if np.iscomplexobj(state):
        return np.real(np.vdot(state, H @ state))
    else:
        return np.dot(state, H @ state)


def _objective(params, n, coupling, field, layers, H, E0):
    """Objective: mean energy gap per spin = (E_circuit - E0) / n."""
    state = _circuit_state(params, n, layers)
    E = _energy(state, H)
    return (E - E0) / n


def _objective_with_grad(params, n, coupling, field, layers, H, E0):
    """Objective with numerical gradient."""
    val = _objective(params, n, coupling, field, layers, H, E0)
    return val


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 130.0  # 10s safety margin before 140s (evaluator kills at 150s)
    
    # Build Hamiltonian and find exact ground state
    H = _build_hamiltonian(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    E0 = eigvals[0]
    
    # Initial guess: all zeros (product |0> state) or small random
    best_params = np.zeros(layers * n)
    best_val = _objective(best_params, n, coupling, field, layers, H, E0)
    
    # Try multiple initializations
    np.random.seed(42)
    
    n_restarts = 8 if n == 4 else 4
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            params0 = np.zeros(layers * n)
        elif restart == 1:
            params0 = np.pi / 2 * np.ones(layers * n)
        elif restart == 2:
            params0 = np.random.uniform(-np.pi/4, np.pi/4, size=(layers * n))
        else:
            params0 = np.random.uniform(-np.pi, np.pi, size=(layers * n))
        
        # Use L-BFGS-B with numerical gradient
        try:
            result = minimize(
                _objective_with_grad,
                params0,
                args=(n, coupling, field, layers, H, E0),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if result.success or result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 15.0:
            break
    
    # Final polish: try a tighter optimization from the best point
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _objective_with_grad,
                best_params,
                args=(n, coupling, field, layers, H, E0),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Reshape to (layers, n)
    return best_params.reshape(layers, n)

# EVOLVE-BLOCK-END
