"""Optimized quantum circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    H = np.zeros((2**n, 2**n), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        op = np.array([[1]], dtype=complex)
        for j in range(n):
            if j == i or j == i + 1:
                op = np.kron(op, Z)
            else:
                op = np.kron(op, I)
        H += -coupling * op
    
    # X terms: -field * X_i
    for i in range(n):
        op = np.array([[1]], dtype=complex)
        for j in range(n):
            if j == i:
                op = np.kron(op, X)
            else:
                op = np.kron(op, I)
        H += -field * op
    
    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0])


def _circuit_state_vector(n, theta, layers):
    """
    Build the state vector from the circuit.
    theta: (layers, n) array of Ry angles
    Circuit: |0>^n, then for each layer l:
      - Apply Ry(theta[l, i]) to qubit i for all i
      - If not last layer, apply CZ on adjacent pairs (i, i+1) for i in ascending order
    """
    dim = 2**n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    # Precompute bit masks for each qubit
    bit_masks = [(1 << (n - 1 - i)) for i in range(n)]
    
    for l in range(layers):
        # Apply Ry(theta) to each qubit
        for i in range(n):
            theta_i = theta[l, i]
            c = np.cos(theta_i / 2)
            s = np.sin(theta_i / 2)
            mask = bit_masks[i]
            
            # Vectorized: separate indices where bit i is 0 vs 1
            # For bit i = 0: |0> -> c|0> + s|1>
            # For bit i = 1: |1> -> -s|0> + c|1>
            new_state = np.zeros_like(state)
            
            # Indices where bit i is 0
            mask0 = ~np.arange(dim) & mask
            idx0 = np.where(mask0)[0]
            idx1 = idx0 | mask
            
            # Apply Ry
            new_state[idx0] += c * state[idx0]
            new_state[idx1] += s * state[idx0]
            new_state[idx0] += -s * state[idx1]
            new_state[idx1] += c * state[idx1]
            
            state = new_state
        
        # Apply CZ on adjacent pairs if not last layer
        if l < layers - 1:
            for i in range(n - 1):
                mask_i = bit_masks[i]
                mask_ip1 = bit_masks[i + 1]
                # Find indices where both bits are 1
                mask_both = mask_i & mask_ip1
                idx_both = np.where((np.arange(dim) & mask_both) == mask_both)[0]
                state[idx_both] *= -1
    
    return state


def _circuit_energy(n, coupling, field, theta, layers):
    """Compute the energy of the circuit state."""
    H = _build_hamiltonian(n, coupling, field)
    state = _circuit_state_vector(n, theta, layers)
    energy = np.vdot(state, H @ state).real
    return float(energy)


def _objective(params, n, coupling, field, layers, ground_energy):
    """Objective function for optimization."""
    theta = params.reshape((layers, n))
    energy = _circuit_energy(n, coupling, field, theta, layers)
    return (energy - ground_energy) / n


def angles(n, coupling, field, layers):
    """Optimize circuit angles to approximate the ground state."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Precompute ground energy and Hamiltonian
    ground_energy = _exact_ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    
    def fast_objective(params):
        """Faster objective using precomputed H."""
        theta = params.reshape((layers, n))
        state = _circuit_state_vector(n, theta, layers)
        energy = np.vdot(state, H @ state).real
        return (energy - ground_energy) / n
    
    # Initial guess
    best_params = np.zeros(layers * n)
    best_score = float('inf')
    
    # Generate diverse initializations based on physics
    initializations = []
    
    # 1. All zeros
    initializations.append(np.zeros(layers * n))
    
    # 2. All pi/4
    initializations.append(np.full(layers * n, np.pi / 4))
    
    # 3. Physics-informed: transverse field Ising ground state
    # At field=0: product |0> or |1>, theta=0
    # At large field: |+> states, theta=pi/2
    # The crossover is around field~1
    # Use tanh-like interpolation
    init_physics = np.zeros(layers * n)
    for l in range(layers):
        for i in range(n):
            # Edge qubits feel less coupling, so more influenced by field
            edge_factor = 1.0 if i == 0 or i == n-1 else 0.8
            theta_val = np.pi / 2 * np.tanh(field * edge_factor)
            init_physics[l * n + i] = theta_val
    initializations.append(init_physics)
    
    # 4. Layer-dependent physics: earlier layers more entangling, later more local
    init_layer = np.zeros(layers * n)
    for l in range(layers):
        layer_weight = 1.0 - 0.2 * l  # Decrease with layer
        for i in range(n):
            edge_factor = 1.0 if i == 0 or i == n-1 else 0.8
            theta_val = np.pi / 2 * np.tanh(field * edge_factor) * layer_weight
            init_layer[l * n + i] = theta_val
    initializations.append(init_layer)
    
    # 5-8. Random initializations with different seeds
    for seed in [42, 123, 7, 999]:
        np.random.seed(seed)
        initializations.append(np.random.uniform(-np.pi/2, np.pi/2, layers * n))
    
    # 9. Perturbed physics init
    np.random.seed(7)
    initializations.append(init_physics + np.random.normal(0, 0.15, layers * n))
    
    # 10. Anti-symmetric pattern (common in Ising ground states)
    init_asym = np.zeros(layers * n)
    for l in range(layers):
        for i in range(n):
            sign = 1.0 if i % 2 == 0 else -1.0
            init_asym[l * n + i] = sign * np.pi / 4 * np.tanh(field)
    initializations.append(init_asym)
    
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    for init in initializations:
        if time.time() > deadline - 15:
            break
        
        try:
            result = minimize(
                fast_objective,
                init,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 200,
                    'maxfun': 800,
                    'ftol': 1e-12,
                    'gtol': 1e-8
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5:
        try:
            result2 = minimize(
                fast_objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 400,
                    'maxfun': 1500,
                    'ftol': 1e-14,
                    'gtol': 1e-10
                }
            )
            
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass
    
    theta = best_params.reshape((layers, n))
    return theta
# EVOLVE-BLOCK-END
