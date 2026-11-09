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


def _apply_ry(state, n, qubit, theta):
    """Apply Ry(theta) to qubit in-place on state vector."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    # For each basis state, if bit qubit is 0: new += c*old, if 1: new += s*old for |1> component
    # We need to be careful with the bit ordering
    # qubit i corresponds to bit position (n-1-i) in the integer representation
    # Actually, let's use a different approach: iterate over all states
    
    new_state = np.zeros_like(state)
    for idx in range(2**n):
        bits = [(idx >> (n-1-k)) & 1 for k in range(n)]
        if bits[qubit] == 0:
            # Ry maps |0> -> c|0> + s|1>
            idx1 = idx | (1 << (n-1-qubit))  # set bit qubit to 1
            new_state[idx] += c * state[idx]
            new_state[idx1] += s * state[idx]
        else:
            # Ry maps |1> -> -s|0> + c|1>
            idx0 = idx & ~(1 << (n-1-qubit))  # clear bit qubit
            new_state[idx] += c * state[idx]
            new_state[idx0] += -s * state[idx]
    return new_state


def _apply_cz(state, n, qubit_a, qubit_b):
    """Apply CZ to qubits a and b in-place on state vector."""
    new_state = state.copy()
    for idx in range(2**n):
        bits = [(idx >> (n-1-k)) & 1 for k in range(n)]
        if bits[qubit_a] == 1 and bits[qubit_b] == 1:
            new_state[idx] = -state[idx]
    return new_state


def _circuit_state_vector(n, theta, layers):
    """
    Build the state vector from the circuit.
    theta: (layers, n) array of Ry angles
    Circuit: |0>^n, then for each layer l:
      - Apply Ry(theta[l, i]) to qubit i for all i
      - If not last layer, apply CZ on adjacent pairs (i, i+1) for i in ascending order
    """
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    for l in range(layers):
        # Apply Ry(theta) to each qubit
        for i in range(n):
            state = _apply_ry(state, n, i, theta[l, i])
        
        # Apply CZ on adjacent pairs if not last layer
        if l < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, n, i, i + 1)
    
    return state


def _circuit_energy(n, coupling, field, theta, layers, H=None):
    """Compute the energy of the circuit state."""
    if H is None:
        H = _build_hamiltonian(n, coupling, field)
    state = _circuit_state_vector(n, theta, layers)
    energy = np.vdot(state, H @ state).real
    return float(energy)


def _objective(params, n, coupling, field, layers, ground_energy, H):
    """Objective function for optimization."""
    theta = params.reshape((layers, n))
    energy = _circuit_energy(n, coupling, field, theta, layers, H)
    return (energy - ground_energy) / n


def angles(n, coupling, field, layers):
    """Optimize circuit angles to approximate the ground state."""
    start_time = time.time()
    deadline = start_time + 140  # Leave safety margin before 150s kill
    
    # Precompute ground energy and Hamiltonian
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _exact_ground_energy(n, coupling, field)
    
    best_params = np.zeros(layers * n)
    best_score = float('inf')
    
    # Try multiple initializations with different strategies
    initializations = []
    
    # 1. All zeros
    initializations.append(np.zeros(layers * n))
    
    # 2. All pi/4
    initializations.append(np.full(layers * n, np.pi / 4))
    
    # 3. Last layer pi/2
    init3 = np.zeros(layers * n)
    init3[-n:] = np.pi / 2
    initializations.append(init3)
    
    # 4. Physics-based interpolation
    init4 = np.zeros(layers * n)
    for l in range(layers):
        for i in range(n):
            init4[l * n + i] = np.pi / 2 * min(1.0, field)
    initializations.append(init4)
    
    # 5-8. Random initializations with different seeds
    for seed in [42, 123, 7, 99]:
        np.random.seed(seed)
        initializations.append(np.random.uniform(-np.pi/2, np.pi/2, layers * n))
    
    # 9. Symmetric initialization based on field strength
    init9 = np.zeros(layers * n)
    # For small field, ground state is close to |0>^n, so angles near 0
    # For large field, ground state is close to |+>^n, so angles near pi/2
    # Middle layers can have different angles
    base_angle = np.arctan(field / coupling) if coupling > 0 else np.pi/2
    for l in range(layers):
        scale = 1.0 - 0.3 * l  # Decrease angle in later layers
        for i in range(n):
            init9[l * n + i] = base_angle * scale
    initializations.append(init9)
    
    # 10. Another physics-inspired init
    init10 = np.zeros(layers * n)
    for l in range(layers):
        for i in range(n):
            # Alternate pattern based on position
            pos_factor = 0.8 + 0.2 * np.sin(np.pi * i / max(1, n-1))
            init10[l * n + i] = (np.pi / 2) * min(1.0, field) * pos_factor * (1.0 - 0.2*l)
    initializations.append(init10)
    
    # Run Nelder-Mead from each initialization
    for idx, init in enumerate(initializations):
        if time.time() > deadline - 5:
            break
        
        try:
            result = minimize(
                _objective,
                init,
                args=(n, coupling, field, layers, ground_energy, H),
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'maxfev': 800,
                    'xatol': 1e-7,
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    # Refine with L-BFGS-B from the best result
    if time.time() < deadline - 10:
        try:
            result2 = minimize(
                _objective,
                best_params,
                args=(n, coupling, field, layers, ground_energy, H),
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={
                    'maxiter': 300,
                    'maxfun': 1000,
                    'ftol': 1e-12,
                    'gtol': 1e-8
                }
            )
            
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass
    
    # Try a few more local refinements from slightly perturbed best params
    if time.time() < deadline - 5:
        for pert in [0.01, 0.05, 0.1]:
            if time.time() > deadline - 3:
                break
            try:
                np.random.seed(int(time.time() * 1000) % 10000)
                perturbed = best_params + np.random.uniform(-pert, pert, layers * n)
                result3 = minimize(
                    _objective,
                    perturbed,
                    args=(n, coupling, field, layers, ground_energy, H),
                    method='L-BFGS-B',
                    bounds=[(-np.pi, np.pi)] * (layers * n),
                    options={
                        'maxiter': 200,
                        'maxfun': 500,
                        'ftol': 1e-12,
                        'gtol': 1e-8
                    }
                )
                
                if result3.fun < best_score:
                    best_score = result3.fun
                    best_params = result3.x.copy()
            except Exception:
                continue
    
    theta = best_params.reshape((layers, n))
    return theta
# EVOLVE-BLOCK-END
