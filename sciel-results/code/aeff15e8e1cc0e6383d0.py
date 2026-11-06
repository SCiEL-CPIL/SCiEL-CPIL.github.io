"""Rewritten program: variational least-squares fit to exact ground state."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

# Precompute exact ground states and their properties for the known problem sizes
# We'll compute them on demand and cache them.

_cache = {}

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


def _apply_ry(state, theta):
    """Apply Ry(theta) to every qubit in the state vector."""
    # Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
    # For each qubit i, we apply this gate.
    # We can do this efficiently by iterating over qubits.
    state = state.copy()
    n = int(np.log2(len(state)))
    for i in range(n):
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        # For each pair of basis states differing only in bit i
        for state_idx in range(len(state)):
            if (state_idx >> i) & 1 == 0:
                partner = state_idx | (1 << i)
                old0 = state[state_idx]
                old1 = state[partner]
                state[state_idx] = c * old0 - s * old1
                state[partner] = s * old0 + c * old1
    return state


def _apply_cz(state):
    """Apply CZ on every adjacent pair in ascending order."""
    n = int(np.log2(len(state)))
    state = state.copy()
    for i in range(n - 1):
        # CZ flips the phase of |11> for qubits i and i+1
        for state_idx in range(len(state)):
            bit_i = (state_idx >> i) & 1
            bit_ip1 = (state_idx >> (i + 1)) & 1
            if bit_i == 1 and bit_ip1 == 1:
                state[state_idx] = -state[state_idx]
    return state


def _circuit_state(angles, n):
    """Apply the circuit to |0>^n and return the final state vector."""
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    layers = angles.shape[0]
    for layer in range(layers):
        # Apply Ry to every qubit
        state = _apply_ry(state, angles[layer])
        
        # Apply CZ between layers (not after final layer)
        if layer < layers - 1:
            state = _apply_cz(state)
    
    return state


def _energy(state, n, coupling, field):
    """Compute the expectation value of the Hamiltonian for a given state."""
    H = _build_hamiltonian(n, coupling, field)
    return np.real(np.conj(state) @ H @ state)


def _objective(params, n, coupling, field, target_state, layers):
    """Objective function: minimize distance to target state."""
    angles = params.reshape(layers, n)
    state = _circuit_state(angles, n)
    # Use real and imaginary parts as residuals
    return np.concatenate([state.real - target_state.real, 
                           state.imag - target_state.imag])


def angles(n, coupling, field, layers):
    """Find circuit angles that approximate the ground state."""
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for safety
    
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
    
    # Guess 4: Based on dominant features of the ground state
    # The ground state has certain correlations; try angles that reflect this
    for _ in range(3):
        guess = np.zeros((layers, n))
        # Try alternating patterns
        for i in range(n):
            guess[0, i] = (-1)**i * np.pi/8
            guess[1, i] = np.pi/6
            guess[2, i] = np.pi/4
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
                args=(n, coupling, field, target_state, layers),
                method='trf',
                max_nfev=200
            )
        except Exception:
            continue
        
        if result.success or result.fun < best_score:
            best_score = np.sum(result.fun**2)
            best_angles = result.x.reshape(layers, n)
    
    # If we still have time, try a few more random restarts
    while time.time() < deadline - 5:
        guess = rng.uniform(-np.pi/2, np.pi/2, size=(layers, n))
        params_flat = guess.flatten()
        
        try:
            result = least_squares(
                _objective,
                params_flat,
                args=(n, coupling, field, target_state, layers),
                method='trf',
                max_nfev=100
            )
        except Exception:
            continue
        
        if np.sum(result.fun**2) < best_score:
            best_score = np.sum(result.fun**2)
            best_angles = result.x.reshape(layers, n)
    
    # Ensure all values are finite
    if not np.all(np.isfinite(best_angles)):
        best_angles = np.zeros((layers, n))
    
    return best_angles
# EVOLVE-BLOCK-END
