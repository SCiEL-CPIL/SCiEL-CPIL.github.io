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
    """Apply Ry(theta) to qubit in-place on state vector using bit manipulation."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    dim = 2**n
    new_state = np.zeros_like(state)
    bit_mask = 1 << (n - 1 - qubit)
    for idx in range(dim):
        if (idx & bit_mask) == 0:
            idx1 = idx | bit_mask
            new_state[idx] += c * state[idx]
            new_state[idx1] += s * state[idx]
        else:
            idx0 = idx & ~bit_mask
            new_state[idx0] += s * state[idx]
            new_state[idx] += c * state[idx]
    return new_state


def _apply_cz(state, n, qubit1, qubit2):
    """Apply CZ gate between qubit1 and qubit2 using bit manipulation."""
    dim = 2**n
    mask1 = 1 << (n - 1 - qubit1)
    mask2 = 1 << (n - 1 - qubit2)
    new_state = state.copy()
    for idx in range(dim):
        if (idx & mask1) != 0 and (idx & mask2) != 0:
            new_state[idx] = -state[idx]
    return new_state


def _circuit_state_vector(n, theta, layers):
    """
    Build the state vector from the circuit using efficient bit manipulation.
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


def _objective_and_grad(params, n, coupling, field, layers, ground_energy, H):
    """Objective function and gradient for optimization."""
    theta = params.reshape((layers, n))
    energy = _circuit_energy(n, coupling, field, theta, layers, H)
    obj = (energy - ground_energy) / n
    
    # Finite difference gradient
    grad = np.zeros_like(params)
    h = 1e-7
    for k in range(len(params)):
        params_plus = params.copy()
        params_plus[k] += h
        e_plus = _circuit_energy(n, coupling, field, params_plus.reshape((layers, n)), layers, H)
        params_minus = params.copy()
        params_minus[k] -= h
        e_minus = _circuit_energy(n, coupling, field, params_minus.reshape((layers, n)), layers, H)
        grad[k] = (e_plus - e_minus) / (2 * h) / n
    
    return obj, grad


def angles(n, coupling, field, layers):
    """Optimize circuit angles to approximate the ground state."""
    start_time = time.time()
    deadline = start_time + 110  # Leave safety margin
    
    # Precompute ground energy and Hamiltonian
    ground_energy = _exact_ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    
    # Physics-based initializations
    # For transverse-field Ising:
    # - Small field: ground state close to |0>^n or |1>^n, small angles
    # - Large field: ground state close to |+>^n, angles near pi/2
    # - Intermediate: mixture
    
    initializations = []
    
    # 1. All zeros
    initializations.append(np.zeros(layers * n))
    
    # 2. All pi/2 (maximally transverse)
    initializations.append(np.full(layers * n, np.pi / 2))
    
    # 3. Physics-based interpolation
    # At field=0, optimal is theta=0; at large field, theta=pi/2
    # Use a smooth interpolation
    f_scale = min(field, 2.0) / 2.0  # normalize field
    base_angle = np.pi / 2 * f_scale
    init3 = np.full(layers * n, base_angle)
    initializations.append(init3)
    
    # 4. Alternating pattern (for n=4,6)
    init4 = np.zeros(layers * n)
    for l in range(layers):
        for i in range(n):
            init4[l * n + i] = np.pi / 2 * (1 if i % 2 == 0 else 0.5) * min(1.0, field)
    initializations.append(init4)
    
    # 5. Random initializations
    np.random.seed(42)
    initializations.append(np.random.uniform(-np.pi, np.pi, layers * n))
    np.random.seed(123)
    initializations.append(np.random.uniform(-np.pi, np.pi, layers * n))
    np.random.seed(456)
    initializations.append(np.random.uniform(-np.pi/2, np.pi/2, layers * n))
    
    best_params = initializations[0].copy()
    best_score = _circuit_energy(n, coupling, field, best_params.reshape((layers, n)), layers, H) - ground_energy
    best_score = best_score / n
    
    for init in initializations:
        if time.time() > deadline - 10:
            break
        
        try:
            # Use L-BFGS-B with analytic (finite-difference) gradient
            result = minimize(
                lambda p: _objective_and_grad(p, n, coupling, field, layers, ground_energy, H)[0],
                init,
                jac=lambda p: _objective_and_grad(p, n, coupling, field, layers, ground_energy, H)[1],
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={
                    'maxiter': 300,
                    'maxfun': 2000,
                    'ftol': 1e-12,
                    'gtol': 1e-8
                }
            )
            
            current_score = result.fun
            if current_score < best_score:
                best_score = current_score
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    # Final polish with more iterations if time permits
    if time.time() < deadline - 20:
        try:
            result2 = minimize(
                lambda p: _objective_and_grad(p, n, coupling, field, layers, ground_energy, H)[0],
                best_params,
                jac=lambda p: _objective_and_grad(p, n, coupling, field, layers, ground_energy, H)[1],
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={
                    'maxiter': 500,
                    'maxfun': 5000,
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
