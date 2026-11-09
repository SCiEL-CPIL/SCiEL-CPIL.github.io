"""Optimize 3-layer Ry/CZ circuit for Ising chain ground states."""
import numpy as np
import time
from scipy.optimize import minimize

# Precompute exact ground energies for all test cases
# H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
# Open chain, qubit i is bit i

def _build_hamiltonian(n, coupling, field):
    """Build Hamiltonian matrix for open transverse-field Ising chain."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            z_val = 1 if bit_i == bit_ip1 else -1
            H[state, state] += -coupling * z_val
    
    # X_i terms
    for i in range(n):
        for state in range(dim):
            # X flips bit i
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    
    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via eigendecomposition."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real


# Cache exact ground energies
_ground_energy_cache = {}

def _get_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key not in _ground_energy_cache:
        _ground_energy_cache[key] = _exact_ground_energy(n, coupling, field)
    return _ground_energy_cache[key]


def _circuit_state_vector(theta, n, layers):
    """
    Apply the circuit to |0>^n and return the state vector.
    theta: (layers, n) array of Ry angles
    Between layers, apply CZ on adjacent pairs in ascending order.
    No entangler after final layer.
    """
    dim = 2 ** n
    # Start with |0>^n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    # Ry rotation matrix
    def ry(theta_val):
        c = np.cos(theta_val / 2)
        s = np.sin(theta_val / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    
    # CZ gate
    def cz_gate(state_vec, n_qubits, i, j):
        """Apply CZ between qubits i and j (ascending order means i<j)."""
        new_state = state_vec.copy()
        for state in range(2 ** n_qubits):
            bit_i = (state >> i) & 1
            bit_j = (state >> j) & 1
            if bit_i == 1 and bit_j == 1:
                new_state[state] = -state_vec[state]
            else:
                new_state[state] = state_vec[state]
        return new_state
    
    for layer in range(layers):
        # Apply Ry(theta[layer, i]) to qubit i for all i
        for i in range(n):
            theta_val = theta[layer, i]
            c = np.cos(theta_val / 2)
            s = np.sin(theta_val / 2)
            # Apply Ry to qubit i
            new_state = np.zeros(dim, dtype=np.complex128)
            for state in range(dim):
                bit_i = (state >> i) & 1
                if bit_i == 0:
                    # |0> -> c|0> + s|1>
                    new_state[state] += c * state[state]
                    new_state[state | (1 << i)] += s * state[state]
                else:
                    # |1> -> -s|0> + c|1>
                    new_state[state & ~(1 << i)] += -s * state[state]
                    new_state[state] += c * state[state]
            state = new_state
        
        # Apply CZ gates between adjacent pairs in ascending order (if not last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                state = cz_gate(state, n, i, i + 1)
    
    return state


def _circuit_energy(state, n, coupling, field):
    """Compute <psi|H|psi> for the circuit state."""
    H = _build_hamiltonian(n, coupling, field)
    energy = np.vdot(state, H @ state).real
    return energy


def _objective(theta_flat, n, coupling, field, layers):
    """Objective function: mean energy gap per spin."""
    theta = theta_flat.reshape(layers, n)
    state = _circuit_state_vector(theta, n, layers)
    energy = _circuit_energy(state, n, coupling, field)
    ground = _get_ground_energy(n, coupling, field)
    gap = (energy - ground) / n
    return gap


# Test cases
TEST_CASES = [
    (4, 1.0, 0.45),
    (4, 1.0, 1.0),
    (4, 1.0, 1.65),
    (6, 1.0, 0.45),
    (6, 1.0, 1.0),
    (6, 1.0, 1.65),
]

LAYERS = 3

def _optimize_single(n, coupling, field, deadline):
    """Optimize parameters for a single case with time limit."""
    n_params = LAYERS * n
    
    best_theta = None
    best_gap = np.inf
    
    # Multi-restart Nelder-Mead
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Initialize
        if restart == 0:
            x0 = np.zeros(n_params)
        else:
            x0 = np.random.uniform(-np.pi, np.pi, n_params)
        
        try:
            result = minimize(
                _objective,
                x0,
                args=(n, coupling, field, LAYERS),
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True,
                }
            )
            
            gap = result.fun
            if gap < best_gap:
                best_gap = gap
                best_theta = result.x
        
        except Exception:
            continue
    
    if best_theta is None:
        best_theta = np.zeros(n_params)
    
    return best_theta.reshape(LAYERS, n), best_gap


def angles(n, coupling, field, layers):
    """
    Return optimized circuit parameters for the given Ising chain.
    
    Args:
        n: number of qubits (4 or 6)
        coupling: Ising coupling strength
        field: transverse field strength
        layers: number of layers (always 3)
    
    Returns:
        (layers, n) array of Ry angles in radians
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Precompute ground energy
    ground = _get_ground_energy(n, coupling, field)
    
    # Optimize
    theta, gap = _optimize_single(n, coupling, field, deadline)
    
    return theta
