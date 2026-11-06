import numpy as np
from scipy.optimize import minimize
import time

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling*sum Z_i Z_(i+1) - field*sum X_i."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Z_i Z_(i+1) terms (diagonal)
    for i in range(n - 1):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            H[idx, idx] += -coupling * zi * zip1
    
    # X_i terms (off-diagonal)
    for i in range(n):
        for idx in range(dim):
            flipped = idx ^ (1 << i)
            H[idx, flipped] += -field
    
    return H

def _get_ground_energy_and_vector(n, coupling, field):
    """Compute exact ground state energy and vector by diagonalizing H."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    ground_energy = eigvals[0]
    ground_vec = eigvecs[:, 0]
    return ground_energy, ground_vec

def _apply_ry_vectorized(state, theta, qubit, n):
    """Apply Ry(theta) to qubit using vectorized operations."""
    dim = 2 ** n
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    
    # Create index arrays
    mask = 1 << qubit
    idx0 = np.arange(dim)
    
    # Separate indices where qubit is 0 or 1
    is_0 = (idx0 & mask) == 0
    is_1 = (idx0 & mask) == mask
    
    result = np.zeros_like(state)
    
    # For indices where qubit is 0
    idx0_0 = idx0[is_0]
    idx1_0 = idx0_0 | mask
    v0 = state[idx0_0]
    v1 = state[idx1_0]
    result[idx0_0] = c * v0 - s * v1
    result[idx1_0] = s * v0 + c * v1
    
    return result

def _apply_cz_vectorized(state, q1, q2, n):
    """Apply CZ gate between qubits q1 and q2 using vectorized operations."""
    dim = 2 ** n
    mask1 = 1 << q1
    mask2 = 1 << q2
    
    idx = np.arange(dim)
    both_1 = ((idx & mask1) == mask1) & ((idx & mask2) == mask2)
    
    result = state.copy()
    result[both_1] *= -1.0
    
    return result

def _build_circuit_state_fast(theta, n, layers):
    """Build the state vector from the circuit parameters using vectorized gates."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            state = _apply_ry_vectorized(state, theta[layer, q], q, n)
        # Apply CZ between this layer and next (if not last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz_vectorized(state, q, q + 1, n)
    
    return state

def _compute_energy_fast(state, H):
    """Compute expectation value using pre-built Hamiltonian."""
    return np.real(np.vdot(state, H @ state))

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Get ground energy and Hamiltonian
    ground_energy, ground_vec = _get_ground_energy_and_vector(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    
    def objective(params):
        theta = params.reshape((layers, n))
        state = _build_circuit_state_fast(theta, n, layers)
        energy = _compute_energy_fast(state, H)
        return (energy - ground_energy) / n
    
    # Initial guess: try multiple different starting points
    best_result = None
    best_val = np.inf
    
    # Generate diverse starting points
    start_points = []
    
    # Starting point 1: all zeros
    start_points.append(np.zeros(layers * n))
    
    # Starting point 2: pi/2 in last layer
    x0_2 = np.zeros(layers * n)
    x0_2[(layers-1)*n:(layers)*n] = np.pi / 2
    start_points.append(x0_2)
    
    # Starting point 3: pi/4 in all layers
    x0_3 = np.full(layers * n, np.pi / 4)
    start_points.append(x0_3)
    
    # Starting point 4: -pi/4 in all layers
    x0_4 = np.full(layers * n, -np.pi / 4)
    start_points.append(x0_4)
    
    # Starting point 5: alternating pi/4, -pi/4
    x0_5 = np.zeros(layers * n)
    for i in range(layers * n):
        x0_5[i] = np.pi / 4 if i % 2 == 0 else -np.pi / 4
    start_points.append(x0_5)
    
    # Add random starting points
    np.random.seed(42)
    for _ in range(3):
        start_points.append(np.random.uniform(-np.pi, np.pi, size=layers * n))
    
    # Add small random perturbations
    for _ in range(2):
        start_points.append(np.random.uniform(-0.3, 0.3, size=layers * n))
    
    for x0 in start_points:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                          bounds=[(-np.pi, np.pi)] * (layers * n),
                          options={'maxiter': 3000, 'maxfun': 8000, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: seed solution
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    return best_result.reshape((layers, n))
