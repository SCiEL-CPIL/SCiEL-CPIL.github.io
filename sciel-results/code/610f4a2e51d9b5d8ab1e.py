"""Rewritten program: variational least-squares fit to exact ground state."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

# Precompute exact ground states and their properties for the known problem sizes
# We'll compute them on demand and cache them.

_cache = {}
_cz_phases = {}

def _build_hamiltonian(n, coupling, field):
    """Build the transverse-field Ising Hamiltonian matrix."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            z_i = 1 - 2 * bit_i
            z_ip1 = 1 - 2 * bit_ip1
            H[state, state] += -coupling * z_i * z_ip1
    
    # X terms: -field * sum X_i
    for i in range(n):
        for state in range(dim):
            # X_i flips bit i
            flipped = state ^ (1 << i)
            H[flipped, state] += -field
    
    return H


def _get_ground_state(n, coupling, field):
    """Get the exact ground state vector and energy."""
    key = (n, coupling, field)
    if key in _cache:
        return _cache[key]
    
    H = _build_hamiltonian(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    ground_energy = eigvals[0]
    ground_state = eigvecs[:, 0]
    
    # Make the phase deterministic: make the largest-magnitude component real and positive
    max_idx = np.argmax(np.abs(ground_state))
    phase = np.angle(ground_state[max_idx])
    ground_state = ground_state * np.exp(-1j * phase)
    
    _cache[key] = (ground_state, ground_energy)
    return _cache[key]

def _get_cz_phase(n):
    """Precompute the phase vector for the sequence of CZ gates."""
    if n in _cz_phases:
        return _cz_phases[n]
    
    dim = 2 ** n
    phases = np.ones(dim, dtype=float)
    for state in range(dim):
        parity = 0
        for i in range(n - 1):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            if bit_i and bit_ip1:
                parity += 1
        if parity % 2 == 1:
            phases[state] = -1.0
    _cz_phases[n] = phases
    return phases


def _apply_ry_fast(state, theta, n):
    """Apply Ry(theta) to every qubit in the state vector using vectorized ops."""
    # state is flat vector of size 2^n
    # Reshape to (2, 2, ..., 2)
    # We apply Ry to each axis.
    # Ry(theta) = [[c, -s], [s, c]]
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    
    # We need to apply this to each qubit.
    # Iterating over n is fine (n is 4 or 6).
    # For each qubit i, move axis i to 0, apply matrix, move back.
    # To avoid in-place issues with complex numbers and views, we copy or handle carefully.
    # Actually, we can just do it in a loop over qubits, modifying a copy or the original.
    # Since we do this for each layer, and layers are few, it's okay.
    
    state = state.reshape([2] * n)
    
    for i in range(n):
        # Move axis i to 0
        state = np.moveaxis(state, i, 0)
        # Reshape to (2, -1)
        state = state.reshape(2, -1)
        # Apply Ry
        # new_state[0] = c * old_state[0] - s * old_state[1]
        # new_state[1] = s * old_state[0] + c * old_state[1]
        old0 = state[0].copy()
        old1 = state[1].copy()
        state[0] = c * old0 - s * old1
        state[1] = s * old0 + c * old1
        # Reshape back
        state = state.reshape([2] * n)
        # Move axis 0 back to i
        state = np.moveaxis(state, 0, i)
        
    return state.flatten()

def _circuit_state_fast(angles, n):
    """Apply the circuit to |0>^n and return the final state vector."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    layers = angles.shape[0]
    cz_phases = _get_cz_phase(n)
    
    for layer in range(layers):
        # Apply Ry to every qubit
        state = _apply_ry_fast(state, angles[layer], n)
        
        # Apply CZ between layers (not after final layer)
        if layer < layers - 1:
            state = state * cz_phases
    
    return state


def _objective(params, n, target_state, layers):
    """Objective function: minimize distance to target state."""
    angles = params.reshape(layers, n)
    state = _circuit_state_fast(angles, n)
    # Use real and imaginary parts as residuals
    return np.concatenate([state.real - target_state.real, 
                           state.imag - target_state.imag])


def angles(n, coupling, field, layers):
    """Find circuit angles that approximate the ground state."""
    start_time = time.time()
    deadline = start_time + 120  # Leave 30s margin for safety
    
    # Get the exact ground state
    target_state, ground_energy = _get_ground_state(n, coupling, field)
    
    # Initialize with a reasonable guess
    best_angles = np.zeros((layers, n))
    best_score = float('inf')
    
    # Generate multiple initial guesses
    initial_guesses = []
    
    # Guess 1: all zeros
    initial_guesses.append(np.zeros((layers, n)))
    
    # Guess 2: pi/2 in last layer (product |+> state)
    guess2 = np.zeros((layers, n))
    guess2[-1] = np.pi / 2
    initial_guesses.append(guess2)
    
    # Guess 3: small random perturbations
    rng = np.random.default_rng(42)
    for _ in range(5):
        guess = rng.uniform(-np.pi/4, np.pi/4, size=(layers, n))
        initial_guesses.append(guess)
        
    # Try each initial guess with least_squares
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        
        params_flat = guess.flatten()
        
        try:
            result = least_squares(
                _objective,
                params_flat,
                args=(n, target_state, layers),
                method='trf',
                max_nfev=100, # Reduced max evaluations
                ftol=1e-8,
                gtol=1e-8
            )
        except Exception:
            continue
        
        score = np.sum(result.fun**2)
        if score < best_score:
            best_score = score
            best_angles = result.x.reshape(layers, n)
    
    # If we still have time, try a few more random restarts
    while time.time() < deadline - 5:
        guess = rng.uniform(-np.pi/2, np.pi/2, size=(layers, n))
        params_flat = guess.flatten()
        
        try:
            result = least_squares(
                _objective,
                params_flat,
                args=(n, target_state, layers),
                method='trf',
                max_nfev=50, # Even fewer for random restarts
                ftol=1e-8,
                gtol=1e-8
            )
        except Exception:
            continue
        
        score = np.sum(result.fun**2)
        if score < best_score:
            best_score = score
            best_angles = result.x.reshape(layers, n)
    
    # Ensure all values are finite
    if not np.all(np.isfinite(best_angles)):
        best_angles = np.zeros((layers, n))
    
    return best_angles
# EVOLVE-BLOCK-END
