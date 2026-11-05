import numpy as np
from scipy.optimize import minimize

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
    # Determine initial guess based on field strength
    # For small fields, the ground state is closer to |0>^n (angles near 0)
    # For large fields, the ground state is closer to |+>^n (angles near pi/2)
    if field < 0.7:
        init_val = 0.1
    elif field < 1.3:
        init_val = np.pi / 4
    else:
        init_val = np.pi / 2
    
    x0 = np.full((layers, n), init_val)
    x0_flat = x0.flatten()
    
    best_energy = np.inf
    best_angles = x0
    
    # Try the default smart start
    res = minimize(_energy_objective, x0_flat, args=(n, coupling, field, layers),
                   method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
    if res.success and res.fun < best_energy:
        best_energy = res.fun
        best_angles = res.x.reshape(layers, n)
    
    # Try multiple random starts with different seeds for diversity
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        np.random.seed(seed)
        x0_rand = np.random.uniform(-np.pi, np.pi, size=(layers, n))
        x0_rand_flat = x0_rand.flatten()
        
        res = minimize(_energy_objective, x0_rand_flat, args=(n, coupling, field, layers),
                       method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-12})
        
        if res.success and res.fun < best_energy:
            best_energy = res.fun
            best_angles = res.x.reshape(layers, n)
    
    # Also try Nelder-Mead from the best L-BFGS-B result for local refinement
    if best_energy < np.inf:
        res_nm = minimize(_energy_objective, best_angles.flatten(), 
                         args=(n, coupling, field, layers),
                         method='Nelder-Mead', 
                         options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
        if res_nm.success and res_nm.fun < best_energy:
            best_energy = res_nm.fun
            best_angles = res_nm.x.reshape(layers, n)
    
    # Ensure finite
    if not np.all(np.isfinite(best_angles)):
        return np.zeros((layers, n))
    
    return best_angles
