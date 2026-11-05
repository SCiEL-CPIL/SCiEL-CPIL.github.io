import numpy as np
from scipy.optimize import minimize, differential_evolution

def _simulate_circuit(n, angles, layers):
    """Simulate the circuit and return the statevector."""
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0  # |0...0>
    
    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = angles[layer, q]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            new_state = np.zeros_like(state)
            for basis in range(2 ** n):
                bit = (basis >> q) & 1
                if bit == 0:
                    new_state[basis] = c * state[basis]
                    new_state[basis | (1 << q)] = s * state[basis | (1 << q)]
                else:
                    new_state[basis] = -s * state[basis ^ (1 << q)]
                    new_state[basis | (1 << q)] = c * state[basis]
            state = new_state
        
        # Apply CZ between adjacent pairs (except after final layer)
        if layer < layers - 1:
            for q in range(n - 1):
                for basis in range(2 ** n):
                    b1 = (basis >> q) & 1
                    b2 = (basis >> (q + 1)) & 1
                    if b1 == 1 and b2 == 1:
                        state[basis] = -state[basis]
    
    return state

def _compute_energy(state, n, coupling, field):
    """Compute expectation value of H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    energy = 0.0
    probs = np.abs(state) ** 2
    
    # ZZ terms
    for i in range(n - 1):
        zz_sum = 0.0
        for basis in range(2 ** n):
            b1 = (basis >> i) & 1
            b2 = (basis >> (i + 1)) & 1
            z1 = 1 if b1 == 0 else -1
            z2 = 1 if b2 == 0 else -1
            zz_sum += probs[basis] * z1 * z2
        energy += -coupling * zz_sum
    
    # X terms
    for i in range(n):
        x_sum = 0.0
        mask = 1 << i
        for basis in range(2 ** n):
            flipped = basis ^ mask
            x_sum += state[basis].conj() * state[flipped]
        energy += -field * 2 * np.real(x_sum)
    
    return energy

def _energy_objective(angles_flat, n, coupling, field, layers):
    """Flatten angles to 1D for optimizer."""
    angles = angles_flat.reshape(layers, n)
    state = _simulate_circuit(n, angles, layers)
    return _compute_energy(state, n, coupling, field)

def angles(n, coupling, field, layers):
    # Define bounds for each parameter
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    # Use Differential Evolution for global search
    # This is a stochastic global optimizer, distinct from the local L-BFGS-B/Nelder-Mead in the seed
    try:
        res_de = differential_evolution(
            _energy_objective, 
            bounds, 
            args=(n, coupling, field, layers),
            seed=42,
            maxiter=150,
            popsize=15,
            tol=1e-8,
            polish=False # We will polish manually
        )
        best_x = res_de.x
        best_energy = res_de.fun
    except Exception:
        # Fallback to simple random start if DE fails
        best_x = np.zeros(layers * n)
        best_energy = _energy_objective(best_x, n, coupling, field, layers)

    # Polish the result with L-BFGS-B for high precision
    try:
        res_lbfgs = minimize(
            _energy_objective, 
            best_x, 
            args=(n, coupling, field, layers),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14}
        )
        if res_lbfgs.success and res_lbfgs.fun < best_energy:
            best_x = res_lbfgs.x
            best_energy = res_lbfgs.fun
    except Exception:
        pass

    # Reshape and return
    best_angles = best_x.reshape(layers, n)
    
    # Ensure finite
    if not np.all(np.isfinite(best_angles)):
        return np.zeros((layers, n))
    
    return best_angles
