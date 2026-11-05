"""Optimized quantum circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_START_TIME = None
_DEADLINE = 140  # seconds, safety margin before 150s kill

def _check_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) > _DEADLINE


def _build_hamiltonian(n, coupling, field):
    """Build H = -coupling * sum Z_i Z_{i+1} - field * sum X_i"""
    dim = 1 << n
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i + 1] = Z
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H -= coupling * term
    
    # X terms: -field * X_i
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H -= field * term
    
    return H


def _apply_ry(state, n, theta):
    """Apply Ry(theta) to all qubits. theta has shape (n,)"""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    # Ry = [[c, -s], [s, c]]
    # For each qubit, we need to apply this
    for i in range(n):
        # Build the operator for qubit i
        ops = [np.eye(2)] * n
        ops[i] = np.array([[c, -s], [s, c]], dtype=complex)
        U = ops[0]
        for op in ops[1:]:
            U = np.kron(U, op)
        state = U @ state
    return state


def _apply_cz_adjacent(state, n):
    """Apply CZ on every adjacent pair in ascending order"""
    for i in range(n - 1):
        # CZ = |0><0|⊗I + |1><1|⊗|1><1| = diag(1,1,1,-1)
        # For qubits i and i+1
        ops = [np.eye(2)] * n
        # CZ on qubits i and i+1
        cz_2q = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=complex)
        # Reshape to apply on adjacent qubits
        # Easier: build full operator
        for j in range(n):
            if j == i:
                ops[j] = np.array([[1,0],[0,0]], dtype=complex)  # |0><0| part
            elif j == i + 1:
                ops[j] = np.array([[1,0],[0,0]], dtype=complex)  # |0><0| part
            else:
                ops[j] = np.eye(2, dtype=complex)
        
        # Actually, let me just build the full CZ operator directly
        dim = 1 << n
        cz_full = np.eye(dim, dtype=complex)
        # CZ flips sign when both qubits i and i+1 are |1>
        for basis in range(dim):
            bit_i = (basis >> (n - 1 - i)) & 1
            bit_ip1 = (basis >> (n - 1 - i - 1)) & 1
            if bit_i == 1 and bit_ip1 == 1:
                cz_full[basis, basis] = -1
        state = cz_full @ state
    return state


def _circuit_energy(theta, n, coupling, field, H, ground_energy):
    """Compute energy of circuit state minus ground energy, divided by n"""
    # Start from |0>^n
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    layers, nq = theta.shape
    for layer in range(layers):
        state = _apply_ry(state, n, theta[layer])
        if layer < layers - 1:
            state = _apply_cz_adjacent(state, n)
    
    # Compute energy
    energy = np.vdot(state, H @ state).real
    return (energy - ground_energy) / n


def _objective(params, n, coupling, field, H, ground_energy):
    """Objective function for optimization"""
    theta = params.reshape(3, n)
    return _circuit_energy(theta, n, coupling, field, H, ground_energy)


def angles(n, coupling, field, layers):
    global _START_TIME
    _START_TIME = time.time()
    
    # Build Hamiltonian and find ground energy
    H = _build_hamiltonian(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    ground_energy = eigvals[0]
    
    best_theta = np.zeros((layers, n))
    best_energy = np.inf
    
    # Multiple restarts with different initial guesses
    n_restarts = 8
    
    for restart in range(n_restarts):
        if _check_deadline():
            break
        
        # Different initial guesses
        if restart == 0:
            # Seed-like: all zeros except last layer pi/2
            init = np.zeros((layers, n))
            init[-1] = np.pi / 2
        elif restart == 1:
            # Small random
            init = np.random.uniform(-0.5, 0.5, (layers, n))
        elif restart == 2:
            # All pi/2
            init = np.full((layers, n), np.pi / 2)
        elif restart == 3:
            # Gradient-like: increasing
            init = np.linspace(0, np.pi, layers * n).reshape(layers, n)
        elif restart == 4:
            # Alternating
            init = np.zeros((layers, n))
            for i in range(n):
                init[0, i] = 0.3 if i % 2 == 0 else -0.3
                init[1, i] = 0.5 if i % 2 == 0 else -0.5
                init[2, i] = np.pi / 2
        elif restart == 5:
            # Random uniform
            init = np.random.uniform(-np.pi, np.pi, (layers, n))
        elif restart == 6:
            # Symmetric: same theta for all qubits per layer
            init = np.random.uniform(-np.pi/4, np.pi/4, layers)
            init = np.repeat(init[:, None], n, axis=1)
        else:
            # Another random
            init = np.random.uniform(-1, 1, (layers, n))
        
        # Optimize
        try:
            result = opt.minimize(
                _objective,
                init.flatten(),
                args=(n, coupling, field, H, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_theta = result.x.reshape(layers, n)
                
        except Exception:
            continue
    
    return best_theta
# EVOLVE-BLOCK-END
