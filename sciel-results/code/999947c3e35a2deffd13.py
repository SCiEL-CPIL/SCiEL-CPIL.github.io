# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

# Cache for exact ground energies and Hamiltonians
_hamiltonian_cache = {}
_ground_energy_cache = {}

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian matrix for an open chain."""
    key = (n, coupling, field)
    if key in _hamiltonian_cache:
        return _hamiltonian_cache[key]
        
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=float)
    
    for state in range(dim):
        # Diagonal: coupling terms -coupling * sum Z_i Z_{i+1}
        diag_val = 0.0
        for i in range(n - 1):
            zi = 1.0 if (state >> i) & 1 == 0 else -1.0
            zij = 1.0 if (state >> (i + 1)) & 1 == 0 else -1.0
            diag_val -= coupling * zi * zij
        H[state, state] += diag_val
        
        # Off-diagonal: field terms -field * sum X_i
        # X_i flips bit i
        for i in range(n):
            neighbor = state ^ (1 << i)
            H[state, neighbor] -= field
    
    _hamiltonian_cache[key] = H
    return H

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy."""
    key = (n, coupling, field)
    if key in _ground_energy_cache:
        return _ground_energy_cache[key]
    
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues = np.linalg.eigvalsh(H)
    ground_energy = eigenvalues[0]
    _ground_energy_cache[key] = ground_energy
    return ground_energy

def _apply_ry(state, n, i, theta):
    """Apply Ry(theta) to qubit i of the state vector."""
    # Reshape state to isolate qubit i
    # Shape: (2^i, 2, 2^(n-i-1))
    shape = [1] * n
    shape[i] = 2
    # We need to reshape carefully. 
    # Index order in numpy: last axis is fastest.
    # Qubit 0 is LSB (bit 0), Qubit n-1 is MSB (bit n-1).
    # In the flat index `idx`, bit i is `(idx >> i) & 1`.
    # This corresponds to the `i`-th dimension if we reshape to `(2^(n-1-i), ..., 2, ..., 2)`?
    # Let's use a standard reshape.
    # If we reshape to `(2, 2, ..., 2)` with n dimensions, dimension 0 corresponds to MSB?
    # NumPy reshape: `arr.reshape([2]*n)`.
    # Element at index `idx` is `arr[idx >> (n-1), ..., idx & 1]`.
    # So dimension 0 is MSB (qubit n-1), dimension n-1 is LSB (qubit 0).
    # So qubit i is at dimension `n - 1 - i`.
    
    reshaped = state.reshape([2] * n)
    dim_idx = n - 1 - i
    
    # Move the target dimension to the front for easy matrix multiplication
    # axes: [dim_idx, 0, 1, ..., dim_idx-1, dim_idx+1, ...]
    new_axes = [dim_idx] + [j for j in range(n) if j != dim_idx]
    moved = np.transpose(reshaped, new_axes)
    
    # moved shape: (2, 2^(n-1))
    # Ry matrix
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    # Ry = [[c, -s], [s, c]]
    # Note: The state vector is a column vector.
    # new_state = Ry @ old_state
    # In our array, `moved[:, k]` is the vector [amp_0, amp_1] for the k-th configuration of other qubits.
    # new_amp_0 = c * old_amp_0 - s * old_amp_1
    # new_amp_1 = s * old_amp_0 + c * old_amp_1
    
    new_moved = np.empty_like(moved)
    new_moved[0, :] = c * moved[0, :] - s * moved[1, :]
    new_moved[1, :] = s * moved[0, :] + c * moved[1, :]
    
    # Transpose back
    # Inverse of new_axes
    inv_axes = [0] * n
    for k, axis in enumerate(new_axes):
        inv_axes[axis] = k
        
    reshaped = np.transpose(new_moved, inv_axes)
    return reshaped.reshape(state.shape)

def _apply_cz(state, n, i):
    """Apply CZ to qubits i and i+1."""
    reshaped = state.reshape([2] * n)
    dim_i = n - 1 - i
    dim_j = n - 1 - (i + 1)
    
    # Flip sign where both are 1
    # Index in reshaped array for |11> at these positions
    # We can create a mask
    mask = np.zeros([2] * n, dtype=bool)
    # Set mask to True where bit i is 1 and bit i+1 is 1
    # In the reshaped array, axis `dim_i` index 1 and axis `dim_j` index 1.
    # This is tricky to do with a simple slice if dimensions are not adjacent.
    # Easier: iterate over the flat indices? No, that's slow.
    # Use broadcasting.
    
    # Create a view that isolates the two qubits
    # Move dim_i and dim_j to the front
    other_dims = [d for d in range(n) if d not in [dim_i, dim_j]]
    new_axes = [dim_i, dim_j] + other_dims
    moved = np.transpose(reshaped, new_axes)
    
    # moved shape: (2, 2, 2^(n-2))
    # Flip sign of moved[1, 1, :]
    moved[1, 1, :] *= -1.0
    
    # Transpose back
    inv_axes = [0] * n
    for k, axis in enumerate(new_axes):
        inv_axes[axis] = k
        
    reshaped = np.transpose(moved, inv_axes)
    return reshaped.reshape(state.shape)

def _circuit_state_vector(n, params, layers):
    """
    Apply the circuit to |0>^n.
    params shape: (layers, n)
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry(theta) to each qubit
        for i in range(n):
            theta = params[layer, i]
            state = _apply_ry(state, n, i, theta)
        
        # Apply CZ on adjacent pairs (if not last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, n, i)
                
    return state

def _circuit_energy(n, params, coupling, field, layers):
    """Compute the energy of the circuit state."""
    state = _circuit_state_vector(n, params, layers)
    H = _build_hamiltonian(n, coupling, field)
    energy = np.real(np.vdot(state, H @ state))
    return energy

def _objective(params, n, coupling, field, layers, ground_energy, deadline):
    """Objective function: mean energy gap per spin."""
    if time.time() > deadline:
        return float('inf')
    energy = _circuit_energy(n, params, coupling, field, layers)
    gap = (energy - ground_energy) / n
    return gap

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    ground_energy = _exact_ground_energy(n, coupling, field)
    
    best_params = np.zeros((layers, n))
    best_energy = float('inf')
    
    # Initial guesses
    inits = [
        np.zeros((layers, n)),
        np.full((layers, n), np.pi / 4),
        np.full((layers, n), np.pi / 2),
        np.random.uniform(-np.pi, np.pi, (layers, n)),
        np.random.uniform(-np.pi, np.pi, (layers, n)),
    ]
    
    for init in inits:
        if time.time() > deadline - 10:
            break
        
        x0 = init.flatten()
        
        try:
            # Use L-BFGS-B
            # Limit iterations and function calls to avoid long runs
            result = opt.minimize(
                lambda x: _objective(x.reshape(layers, n), n, coupling, field, layers, ground_energy, deadline),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000}
            )
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.reshape(layers, n)
                
        except Exception:
            pass
            
    # Additional random restarts if time permits
    while time.time() < deadline - 5:
        x0 = np.random.uniform(-np.pi, np.pi, (layers, n)).flatten()
        try:
            result = opt.minimize(
                lambda x: _objective(x.reshape(layers, n), n, coupling, field, layers, ground_energy, deadline),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 500}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.reshape(layers, n)
        except Exception:
            pass
            
    return best_params
# EVOLVE-BLOCK-END
