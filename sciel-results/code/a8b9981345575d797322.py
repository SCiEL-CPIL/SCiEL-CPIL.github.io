"""Optimized shallow-circuit Ising ground state approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

_start_time = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _ry_matrix(theta):
    """Return 2x2 Ry rotation matrix."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _apply_ry(state, theta, qubit, n):
    """Apply Ry(theta) to qubit of state (2^n vector). Qubit i is bit i (LSB)."""
    if abs(theta) < 1e-15:
        return state
    shape = [2] * n
    state_r = state.reshape(shape)
    gate = _ry_matrix(theta)
    # Move target qubit axis to position 0 for efficient application
    state_moved = np.moveaxis(state_r, qubit, 0)
    # Apply 2x2 gate on first axis: new_state[a, rest] = sum_b gate[a,b] * state_moved[b, rest]
    state_out = np.tensordot(gate, state_moved, axes=([1], [0]))
    state_out = state_out.reshape(shape)
    return state_out


def _apply_cz(state, q1, q2, n):
    """Apply CZ gate between qubits q1 and q2 (q1 < q2)."""
    shape = [2] * n
    state_r = state.reshape(shape)
    # Move q1 and q2 to positions 0 and 1
    state_moved = np.moveaxis(state_r, [q1, q2], [0, 1])
    # CZ: |11> -> -|11>
    # state_moved has shape (2, 2, 2^(n-2))
    state_moved[1, 1, :] *= -1
    # Move back
    state_out = np.moveaxis(state_moved, [0, 1], [q1, q2])
    return state_out.reshape(-1)


def _circuit_state(theta, n, layers):
    """Build the circuit statevector from angles theta (layers, n)."""
    state = np.zeros(2**n, dtype=np.complex128)
    state[0] = 1.0  # Start in |0>^n
    
    for layer in range(layers):
        # Apply Ry(theta[layer, i]) to each qubit i
        for i in range(n):
            state = _apply_ry(state, theta[layer, i], i, n)
        # Apply CZ on adjacent pairs (except after final layer)
        if layer < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, i, i + 1, n)
    
    return state


def _hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for open transverse-field Ising chain."""
    dim = 2**n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    for basis in range(dim):
        # Z_i Z_{i+1} terms
        for i in range(n - 1):
            bit_i = (basis >> i) & 1
            bit_ip1 = (basis >> (i + 1)) & 1
            zz = 1.0 if bit_i == bit_ip1 else -1.0
            H[basis, basis] -= coupling * zz
        
        # X_i terms
        for i in range(n):
            flipped = basis ^ (1 << i)
            H[basis, flipped] -= field
    
    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy."""
    H = _hamiltonian(n, coupling, field)
    eigenvalues = np.linalg.eigvalsh(H)
    return eigenvalues[0].real


def _circuit_energy(theta, n, coupling, field, layers):
    """Compute the energy expectation value of the circuit state."""
    state = _circuit_state(theta, n, layers)
    H = _hamiltonian(n, coupling, field)
    energy = np.vdot(state, H @ state).real
    return energy


def _objective(params, n, coupling, field, layers, ground_energy):
    """Objective function: mean energy gap per spin."""
    theta = params.reshape(layers, n)
    energy = _circuit_energy(theta, n, coupling, field, layers)
    gap = energy - ground_energy
    return gap / n


def _analytic_init(n, coupling, field, layers):
    """Generate initial guess based on analytical considerations."""
    theta = np.zeros((layers, n))
    # For product states, optimal angle for each qubit to minimize local field
    # tan(theta/2) = field/coupling for interior, different for boundaries
    # Simple heuristic: start with pi/4 for all
    theta[:] = np.pi / 4
    return theta


def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate
    ground states of open transverse-field Ising chains.
    
    Returns: finite real array (layers, n) in radians.
    """
    global _start_time
    
    # Check time
    if time.time() - _start_time > _TIME_LIMIT - 10:
        # Return a reasonable default
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    # Compute exact ground energy
    ground_energy = _exact_ground_energy(n, coupling, field)
    
    # Multi-start optimization
    best_params = None
    best_obj = np.inf
    
    # Generate multiple initial guesses
    inits = []
    
    # 1. Analytical-ish init
    inits.append(_analytic_init(n, coupling, field, layers))
    
    # 2. All zeros
    inits.append(np.zeros((layers, n)))
    
    # 3. Pi/2 on last layer (product |+> state)
    init3 = np.zeros((layers, n))
    init3[-1] = np.pi / 2
    inits.append(init3)
    
    # 4. Random inits
    np.random.seed(42)
    for _ in range(5):
        inits.append(np.random.uniform(-np.pi, np.pi, (layers, n)))
    
    # 5. Staggered inits
    init5 = np.zeros((layers, n))
    for i in range(n):
        init5[:, i] = np.pi / 4 if i % 2 == 0 else -np.pi / 4
    inits.append(init5)
    
    # Optimize from each initial guess
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    for init in inits:
        if time.time() - _start_time > _TIME_LIMIT - 15:
            break
        
        try:
            result = opt.minimize(
                _objective,
                init.flatten(),
                args=(n, coupling, field, layers, ground_energy),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            obj_val = result.fun
            if obj_val < best_obj:
                best_obj = obj_val
                best_params = result.x.reshape(layers, n)
                
        except Exception:
            continue
    
    # If no valid result, return default
    if best_params is None:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    return best_params
# EVOLVE-BLOCK-END
