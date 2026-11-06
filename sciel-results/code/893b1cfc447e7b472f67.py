"""Optimized shallow-circuit Ising ground state approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Cache for precomputed operator data
_OPS_CACHE = {}

def _get_ops(n):
    if n not in _OPS_CACHE:
        dim = 1 << n
        # ZZ signs for each bond
        zz_signs = []
        for i in range(n - 1):
            signs = np.ones(dim)
            for state in range(dim):
                bit_i = (state >> i) & 1
                bit_j = (state >> (i + 1)) & 1
                if bit_i != bit_j:
                    signs[state] = -1.0
            zz_signs.append(signs)
        
        # X flip indices
        x_flip_indices = []
        for i in range(n):
            indices = np.array([state ^ (1 << i) for state in range(dim)])
            x_flip_indices.append(indices)
            
        _OPS_CACHE[n] = (zz_signs, x_flip_indices)
    return _OPS_CACHE[n]

def _circuit_state_fast(theta, n, layers):
    """
    Vectorized circuit state evolution using tensor operations.
    theta: (layers, n)
    """
    dim = 1 << n
    # State as tensor of shape (2,)*n
    state = np.zeros((2,) * n, dtype=complex)
    state[(0,) * n] = 1.0
    
    for l in range(layers):
        # Apply Ry to each qubit
        for i in range(n):
            c = np.cos(theta[l, i] / 2.0)
            s = -np.sin(theta[l, i] / 2.0)
            
            # Move axis i to front
            state = np.moveaxis(state, i, 0)
            # Apply Ry gate: [[c, s], [s, c]]
            # state[0] -> c*state[0] + s*state[1]
            # state[1] -> s*state[0] + c*state[1]
            s0 = state[0]
            s1 = state[1]
            state[0] = c * s0 + s * s1
            state[1] = s * s0 + c * s1
            # Move axis back
            state = np.moveaxis(state, 0, i)
            
        # Apply CZ if not last layer
        if l < layers - 1:
            for i in range(n - 1):
                # CZ on i, i+1
                # Move axes i, i+1 to front
                state = np.moveaxis(state, [i, i+1], [0, 1])
                # Flip sign of |11> component
                state[1, 1] *= -1.0
                # Move axes back
                state = np.moveaxis(state, [0, 1], [i, i+1])
                
    return state.reshape(dim)

def _compute_energy_fast(state, n, coupling, field, zz_signs, x_flip_indices):
    """
    Vectorized energy calculation.
    """
    prob = np.abs(state) ** 2
    energy = 0.0
    
    # ZZ terms
    for i in range(n - 1):
        energy += -coupling * np.dot(prob, zz_signs[i])
        
    # X terms
    for i in range(n):
        flipped_state = state[x_flip_indices[i]]
        energy += -field * np.real(np.dot(np.conj(state), flipped_state))
        
    return energy

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms
    for i in range(n - 1):
        for state_idx in range(dim):
            bit_i = (state_idx >> i) & 1
            bit_j = (state_idx >> (i + 1)) & 1
            zz_val = 1.0 if bit_i == bit_j else -1.0
            H[state_idx, state_idx] += -coupling * zz_val
    
    # X terms
    for i in range(n):
        for state_idx in range(dim):
            flipped = state_idx ^ (1 << i)
            H[state_idx, flipped] += -field
    
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def _objective(params, n, coupling, field, layers, exact_e, zz_signs, x_flip_indices):
    """Objective: mean energy gap per spin = (E_circuit - E_exact) / n"""
    theta = params.reshape(layers, n)
    state = _circuit_state_fast(theta, n, layers)
    e_circuit = _compute_energy_fast(state, n, coupling, field, zz_signs, x_flip_indices)
    return (e_circuit - exact_e) / n

def angles(n, coupling, field, layers):
    """
    Return optimal angles for the circuit.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Precompute operators
    zz_signs, x_flip_indices = _get_ops(n)
    
    # Get exact ground energy
    exact_e = _exact_ground_energy(n, coupling, field)
    
    # Heuristic initial guess
    # For TFIM, ground state aligns with Z (weak field) or X (strong field).
    # Ry(theta) on |0> gives cos(theta/2)|0> - sin(theta/2)|1>.
    # theta=0 -> |0> (Z up), theta=pi -> -|1> (Z down), theta=pi/2 -> |+> (X up).
    # We want to interpolate between 0 and pi/2 (or -pi/2).
    # Using tanh(field) provides a smooth transition.
    base_theta = -np.pi / 2 * np.tanh(field)
    theta0 = np.full((layers, n), base_theta)
    
    starts = [theta0.flatten()]
    
    # Add a small number of random starts to escape local minima
    np.random.seed(42)
    for _ in range(5):
        starts.append(np.random.uniform(-np.pi, np.pi, size=layers * n))
        
    best_params = starts[0]
    best_obj = _objective(best_params, n, coupling, field, layers, exact_e, zz_signs, x_flip_indices)
    
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    for x0 in starts:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10.0:  # Leave 10s buffer
            break
            
        try:
            result = minimize(
                _objective,
                x0,
                args=(n, coupling, field, layers, exact_e, zz_signs, x_flip_indices),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'maxfun': 100, 'ftol': 1e-6}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
            
    return best_params.reshape(layers, n)

# EVOLVE-BLOCK-END
