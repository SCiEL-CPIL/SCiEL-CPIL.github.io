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


def _ry_matrix(theta):
    """Ry(theta) rotation matrix."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, s], [-s, c]], dtype=complex)


def _cz_matrix():
    """CZ gate matrix."""
    return np.diag([1, 1, 1, -1]).astype(complex)


def _circuit_state_vector(n, theta, layers):
    """
    Build the state vector from the circuit.
    theta: (layers, n) array of Ry angles
    Circuit: |0>^n, then for each layer l:
      - Apply Ry(theta[l, i]) to qubit i for all i
      - If not last layer, apply CZ on adjacent pairs (i, i+1) for i in ascending order
    """
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    for l in range(layers):
        # Apply Ry(theta) to each qubit
        for i in range(n):
            state = _apply_single_qubit(state, n, i, _ry_matrix(theta[l, i]))
        
        # Apply CZ on adjacent pairs if not last layer
        if l < layers - 1:
            for i in range(n - 1):
                state = _apply_two_qubit(state, n, i, i + 1, _cz_matrix())
    
    return state


def _apply_single_qubit(state, n, qubit, gate):
    """Apply a single-qubit gate to a specific qubit in the state vector."""
    dim = 2 ** n
    result = np.zeros(dim, dtype=complex)
    bit_pos = n - 1 - qubit
    
    for idx in range(dim):
        bit = (idx >> bit_pos) & 1
        # Partner index: flip the qubit bit
        partner = idx ^ (1 << bit_pos)
        
        if bit == 0:
            # gate[0,0] * |0> + gate[1,0] * |0> -> contributes to |0> and |1>
            result[idx] += gate[0, 0] * state[idx]      # |0> -> |0>
            result[partner] += gate[1, 0] * state[idx]   # |0> -> |1>
        else:
            # gate[0,1] * |1> + gate[1,1] * |1> -> contributes to |0> and |1>
            result[partner] += gate[0, 1] * state[idx]   # |1> -> |0>
            result[idx] += gate[1, 1] * state[idx]       # |1> -> |1>
    
    return result


def _apply_two_qubit(state, n, qubit_a, qubit_b, gate):
    """Apply a two-qubit gate to specific qubits in the state vector."""
    dim = 2 ** n
    result = np.zeros(dim, dtype=complex)
    pos_a = n - 1 - qubit_a
    pos_b = n - 1 - qubit_b
    
    for idx in range(dim):
        bit_a = (idx >> pos_a) & 1
        bit_b = (idx >> pos_b) & 1
        row = bit_a * 2 + bit_b
        
        for col in range(4):
            val = gate[row, col]
            if val == 0:
                continue
            col_a = (col >> 1) & 1
            col_b = col & 1
            
            # Build partner index by replacing bits at pos_a and pos_b
            partner = idx
            if col_a != bit_a:
                partner ^= (1 << pos_a)
            if col_b != bit_b:
                partner ^= (1 << pos_b)
            
            result[partner] += val * state[idx]
    
    return result


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


def _grad_objective(params, n, coupling, field, layers, ground_energy, H, eps=1e-6):
    """Compute gradient using finite differences."""
    m = len(params)
    grad = np.zeros(m)
    for i in range(m):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        f_plus = _objective(params_plus, n, coupling, field, layers, ground_energy, H)
        f_minus = _objective(params_minus, n, coupling, field, layers, ground_energy, H)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def angles(n, coupling, field, layers):
    """Optimize circuit angles to approximate the ground state."""
    start_time = time.time()
    deadline = start_time + 130  # Leave safety margin before 150s kill
    
    # Precompute ground energy and Hamiltonian
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _exact_ground_energy(n, coupling, field)
    
    best_params = np.zeros(layers * n)
    best_score = float('inf')
    
    # Generate diverse initializations
    initializations = []
    
    # 1. All zeros
    initializations.append(np.zeros(layers * n))
    
    # 2. All pi/4
    initializations.append(np.full(layers * n, np.pi / 4))
    
    # 3. Last layer pi/2
    init3 = np.zeros(layers * n)
    init3[-n:] = np.pi / 2
    initializations.append(init3)
    
    # 4. Physics-based: base angle from arctan(field/coupling)
    base_angle = np.arctan(field / coupling) if coupling > 0 else np.pi/2
    init4 = np.zeros(layers * n)
    for l in range(layers):
        scale = 1.0 - 0.2 * l
        for i in range(n):
            init4[l * n + i] = base_angle * scale
    initializations.append(init4)
    
    # 5. Symmetric: all layers same angle
    init5 = np.full(layers * n, np.pi / 4)
    initializations.append(init5)
    
    # 6-11. Random initializations
    for seed in [42, 123, 7, 99, 2024, 555]:
        rng = np.random.RandomState(seed)
        initializations.append(rng.uniform(-np.pi/2, np.pi/2, layers * n))
    
    # 12. Alternating pattern
    init12 = np.zeros(layers * n)
    for l in range(layers):
        for i in range(n):
            sign = 1.0 if i % 2 == 0 else -1.0
            init12[l * n + i] = sign * np.pi / 4 * min(1.0, field)
    initializations.append(init12)
    
    # Run Nelder-Mead from each initialization
    for idx, init in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        
        try:
            result = minimize(
                _objective,
                init,
                args=(n, coupling, field, layers, ground_energy, H),
                method='Nelder-Mead',
                options={
                    'maxiter': 400,
                    'maxfev': 1200,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    # Refine with L-BFGS-B using gradient from the best result
    if time.time() < deadline - 20:
        try:
            result2 = minimize(
                _objective,
                best_params,
                args=(n, coupling, field, layers, ground_energy, H),
                jac=_grad_objective,
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={
                    'maxiter': 500,
                    'maxfun': 2000,
                    'ftol': 1e-14,
                    'gtol': 1e-10
                }
            )
            
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass
    
    # Try additional local refinements from perturbed best params
    if time.time() < deadline - 10:
        for pert in [0.05, 0.1, 0.2]:
            if time.time() > deadline - 5:
                break
            for seed in [1, 2, 3]:
                if time.time() > deadline - 3:
                    break
                try:
                    rng = np.random.RandomState(seed)
                    perturbed = best_params + rng.uniform(-pert, pert, layers * n)
                    result3 = minimize(
                        _objective,
                        perturbed,
                        args=(n, coupling, field, layers, ground_energy, H),
                        jac=_grad_objective,
                        method='L-BFGS-B',
                        bounds=[(-np.pi, np.pi)] * (layers * n),
                        options={
                            'maxiter': 300,
                            'maxfun': 1000,
                            'ftol': 1e-14,
                            'gtol': 1e-10
                        }
                    )
                    
                    if result3.fun < best_score:
                        best_score = result3.fun
                        best_params = result3.x.copy()
                except Exception:
                    continue
    
    # Final refinement
    if time.time() < deadline - 5:
        try:
            result4 = minimize(
                _objective,
                best_params,
                args=(n, coupling, field, layers, ground_energy, H),
                jac=_grad_objective,
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={
                    'maxiter': 500,
                    'maxfun': 2000,
                    'ftol': 1e-15,
                    'gtol': 1e-12
                }
            )
            
            if result4.fun < best_score:
                best_score = result4.fun
                best_params = result4.x.copy()
        except Exception:
            pass
    
    theta = best_params.reshape((layers, n))
    return theta
# EVOLVE-BLOCK-END
