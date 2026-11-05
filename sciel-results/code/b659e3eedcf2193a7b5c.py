"""Optimize shallow Ry/CZ circuit parameters for transverse-field Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from functools import lru_cache

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 115.0  # seconds, safety margin before 120s limit


def _check_deadline():
    return (time.time() - _START_TIME) < _DEADLINE


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        # Z on qubit i and i+1
        zi = np.zeros((dim, dim), dtype=np.complex128)
        for state in range(dim):
            zi[state, state] = 1.0 if (state >> i) & 1 == 0 else -1.0
        # Actually, let's use a cleaner approach with kron products
        
    # Better: build using tensor products
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)
    
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # -coupling * sum_{i=0}^{n-2} Z_i Z_{i+1}
    for i in range(n - 1):
        op = np.array([[1]], dtype=np.complex128)
        for j in range(n - 1, -1, -1):  # from n-1 down to 0
            if j == i:
                op = np.kron(op, Z)
            elif j == i + 1:
                op = np.kron(op, Z)
            else:
                op = np.kron(op, I)
        H -= coupling * op
    
    # -field * sum_i X_i
    for i in range(n):
        op = np.array([[1]], dtype=np.complex128)
        for j in range(n - 1, -1, -1):
            if j == i:
                op = np.kron(op, X)
            else:
                op = np.kron(op, I)
        H -= field * op
    
    return H


def _get_ground_energy(n, coupling, field):
    """Get exact ground state energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real


def _apply_ry(state, theta, qubit, n):
    """Apply Ry(theta) to qubit in the given state (amplitude array)."""
    # state is indexed by bit string where qubit i corresponds to bit i (LSB)
    # Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
    # For qubit in state |0> -> cos(theta/2)|0> + sin(theta/2)|1>
    # For qubit in state |1> -> -sin(theta/2)|0> + cos(theta/2)|1>
    
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    
    new_state = np.zeros_like(state)
    mask = 1 << qubit
    
    for idx in range(len(state)):
        if (idx & mask) == 0:
            # qubit is 0
            new_state[idx] += c * state[idx]
            new_state[idx | mask] += s * state[idx]
        else:
            # qubit is 1
            new_state[idx & ~mask] += -s * state[idx]
            new_state[idx] += c * state[idx]
    
    return new_state


def _apply_cz(state, qubit_a, qubit_b, n):
    """Apply CZ between qubit_a and qubit_b."""
    # CZ: |00>->|00>, |01>->|01>, |10>->|10>, |11>->-|11>
    mask_a = 1 << qubit_a
    mask_b = 1 << qubit_b
    
    new_state = state.copy()
    for idx in range(len(state)):
        if (idx & mask_a) and (idx & mask_b):
            new_state[idx] = -state[idx]
    
    return new_state


def _circuit_state(theta, n, layers):
    """Apply the full circuit to |0>^n and return the state vector."""
    state = np.zeros(2 ** n, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry(theta[layer, i]) to every qubit i
        for i in range(n):
            state = _apply_ry(state, theta[layer, i], i, n)
        
        # Apply CZ on every adjacent pair in ascending order (if not last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, i, i + 1, n)
    
    return state


def _circuit_energy(theta, n, coupling, field, H):
    """Compute <psi|H|psi> for the circuit state."""
    psi = _circuit_state(theta, n, layers)
    Hpsi = H @ psi
    energy = np.vdot(psi, Hpsi).real
    return energy


def _objective(params, n, coupling, field, H, ground_energy):
    """Objective: mean energy gap per spin."""
    theta = params.reshape(layers, n)
    energy = _circuit_energy(theta, n, coupling, field, H)
    gap = energy - ground_energy
    return gap / n


def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    global _START_TIME
    _START_TIME = time.time()
    
    # Build Hamiltonian and get ground energy
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _get_ground_energy(n, coupling, field)
    
    # Initial guess: product |+> state
    theta_init = np.zeros((layers, n))
    theta_init[-1] = np.pi / 2
    
    # Try to optimize using L-BFGS-B
    best_theta = theta_init.copy()
    best_obj = _objective(theta_init.flatten(), n, coupling, field, H, ground_energy)
    
    try:
        result = minimize(
            _objective,
            theta_init.flatten(),
            args=(n, coupling, field, H, ground_energy),
            method='L-BFGS-B',
            options={'maxiter': 500, 'gtol': 1e-6, 'maxls': 50}
        )
        
        if result.fun < best_obj:
            best_theta = result.x.reshape(layers, n)
            best_obj = result.fun
    except Exception:
        pass
    
    # Check deadline
    if not _check_deadline():
        return best_theta
    
    # Try multiple random restarts if time permits
    np.random.seed(42)
    for restart in range(20):
        if not _check_deadline():
            break
        
        # Random initialization
        theta_rand = np.random.uniform(-np.pi, np.pi, (layers, n))
        try:
            result = minimize(
                _objective,
                theta_rand.flatten(),
                args=(n, coupling, field, H, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-6, 'maxls': 50}
            )
            
            if result.fun < best_obj:
                best_theta = result.x.reshape(layers, n)
                best_obj = result.fun
        except Exception:
            pass
    
    return best_theta
# EVOLVE-BLOCK-END
