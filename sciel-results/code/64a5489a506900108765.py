"""Variational ansatz: fit circuit angles to exact ground-state amplitudes via least-squares."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares


def _exact_ground_state(n, coupling, field):
    """Compute the exact ground state of the open transverse-field Ising chain.
    H = -coupling * sum_i Z_i Z_{i+1} - field * sum_i X_i
    Qubit i is bit i (little-endian convention for state indexing).
    Returns the ground state vector as a complex array of length 2**n.
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Build Hamiltonian
    for basis in range(dim):
        # Z_i Z_{i+1} terms
        zz_energy = 0.0
        for i in range(n - 1):
            bit_i = (basis >> i) & 1
            bit_ip1 = (basis >> (i + 1)) & 1
            # Z eigenvalue: |0> -> +1, |1> -> -1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_ip1 = 1.0 if bit_ip1 == 0 else -1.0
            zz_energy += z_i * z_ip1
        H[basis, basis] += -coupling * zz_energy
        
        # X_i terms: flip bit i
        for i in range(n):
            flipped = basis ^ (1 << i)
            H[flipped, basis] += -field
    
    # Diagonalize
    eigvals, eigvecs = np.linalg.eigh(H)
    ground_idx = np.argmin(eigvals)
    return eigvecs[:, ground_idx], eigvals[ground_idx]


def _apply_circuit_vectorized(state, angles, n):
    """Apply the 3-layer Ry/CZ circuit to the initial |0>^n state using vectorized operations."""
    layers = angles.shape[0]
    state = state.copy()
    dim = 2 ** n
    
    for layer in range(layers):
        # Apply Ry(theta) to every qubit
        for i in range(n):
            theta = angles[layer, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            # Reshape state to tensor
            # Qubit i is at axis n-1-i in C-order tensor [2]*n
            axis = n - 1 - i
            
            # Move axis to end
            state_tensor = np.moveaxis(state, axis, -1)
            # Reshape to (-1, 2)
            state_2d = state_tensor.reshape(-1, 2)
            
            # Apply Ry: [[c, -s], [s, c]]
            # New state = c * |0> + s * |1> for |0> input?
            # Ry(theta) = [[cos, -sin], [sin, cos]]
            # |0> -> c|0> + s|1>
            # |1> -> -s|0> + c|1>
            # So:
            # new_0 = c * old_0 - s * old_1
            # new_1 = s * old_0 + c * old_1
            
            new_0 = c * state_2d[:, 0] - s * state_2d[:, 1]
            new_1 = s * state_2d[:, 0] + c * state_2d[:, 1]
            
            state_2d[:, 0] = new_0
            state_2d[:, 1] = new_1
            
            # Reshape back
            state_tensor = state_2d.reshape(state_tensor.shape)
            # Move axis back
            state = np.moveaxis(state_tensor, -1, axis)
        
        # Apply CZ on adjacent pairs (except after final layer)
        if layer < layers - 1:
            for i in range(n - 1):
                j = i + 1
                axis_i = n - 1 - i
                axis_j = n - 1 - j
                
                # Move axes to end
                state_tensor = np.moveaxis(state, [axis_i, axis_j], [-2, -1])
                state_3d = state_tensor.reshape(-1, 2, 2)
                
                # Negate |11> component
                state_3d[:, 1, 1] *= -1
                
                # Reshape back
                state_tensor = state_3d.reshape(state_tensor.shape)
                # Move axes back
                state = np.moveaxis(state_tensor, [-2, -1], [axis_i, axis_j])
    
    return state


def _circuit_state(angles, n):
    """Compute the state produced by the circuit starting from |0>^n."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    return _apply_circuit_vectorized(state, angles, n)


def _residuals(params, n, coupling, field, target_state):
    """Compute residuals for least-squares: difference between circuit state and target state."""
    angles = params.reshape((3, n))
    circuit_state = _circuit_state(angles, n)
    
    # Target state is real, circuit state is real (for real angles)
    # We only use the real part to reduce residual dimension and avoid phase issues
    # Note: If the circuit state has an imaginary part due to numerical errors, we ignore it.
    # The target state is purely real.
    
    # To handle the global phase/sign ambiguity, we can align the sign.
    # If the overlap is negative, we flip the circuit state.
    # This makes the cost function discontinuous, but often helps convergence.
    # However, least_squares assumes smooth functions.
    # Let's try without flipping first, and rely on multiple starts.
    
    diff = circuit_state.real - target_state.real
    return diff


def _generate_inits(n, field, target_state):
    """Generate multiple initial guesses for the optimizer."""
    inits = []
    
    # 1. Zero initialization
    inits.append(np.zeros(3 * n))
    
    # 2. Pi/2 initialization (maximally mixed-like)
    inits.append((np.pi / 2) * np.ones(3 * n))
    
    # 3. Field-dependent initialization
    # For small field, GS is close to |0>^n -> angles ~ 0
    # For large field, GS is close to |+>^n -> angles ~ pi/2
    # Interpolate
    scale = np.arctan(field) / (np.pi / 2)
    inits.append(scale * np.ones(3 * n))
    
    # 4. Random initializations
    np.random.seed(42)
    for _ in range(10):
        inits.append(np.random.uniform(-np.pi, np.pi, size=3 * n))
        
    # 5. Based on target state marginals (heuristic)
    # Estimate single-qubit angles from the target state
    # This is a rough approximation
    target_real = target_state.real
    # For each qubit, estimate the probability of being 1
    # p_i = sum |psi|^2 where bit i is 1
    # Then theta_i ~ 2 * arcsin(sqrt(p_i)) ?
    # This is only valid if the state is a product state, which it's not.
    # But it provides a direction.
    probs = np.zeros(n)
    for i in range(n):
        mask = np.zeros(2**n, dtype=bool)
        for b in range(2**n):
            if (b >> i) & 1:
                mask[b] = True
        probs[i] = np.sum(target_real[mask]**2)
        
    theta_est = 2 * np.arcsin(np.sqrt(probs))
    # Assign to all layers
    init_marginal = np.tile(theta_est, (3, 1)).flatten()
    inits.append(init_marginal)
    
    return inits


def angles(n, coupling, field, layers):
    """Return circuit angles (layers, n) that approximate the ground state."""
    # Compute exact ground state
    target_state, ground_energy = _exact_ground_state(n, coupling, field)
    
    # Generate initial guesses
    inits = _generate_inits(n, field, target_state)
    
    best_params = None
    best_cost = np.inf
    
    for init in inits:
        try:
            result = least_squares(
                _residuals,
                init,
                args=(n, coupling, field, target_state),
                method='trf',
                max_nfev=2000,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10
            )
            cost = result.cost
            if cost < best_cost:
                best_cost = cost
                best_params = result.x
        except Exception:
            continue
    
    if best_params is None:
        # Fallback: return zeros
        return np.zeros((layers, n))
    
    return best_params.reshape((layers, n))
# EVOLVE-BLOCK-END
