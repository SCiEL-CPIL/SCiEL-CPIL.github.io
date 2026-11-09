"""Optimize shallow-circuit Ising ground states with vectorized operations and global optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_state_vectorized(theta, n, layers):
    """Build the circuit state using vectorized operations."""
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    # Precompute bit masks
    bit_masks = [(1 << (n - 1 - i)) for i in range(n)]
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for i in range(n):
            mask = bit_masks[i]
            c = np.cos(theta[layer, i] / 2)
            s = np.sin(theta[layer, i] / 2)
            
            # Vectorized Ry application
            new_state = np.zeros_like(state)
            for basis in range(dim):
                bit_val = (basis >> (n - 1 - i)) & 1
                if bit_val == 0:
                    # |0> -> c|0> + s|1>
                    new_state[basis] += c * state[basis]
                    new_state[basis | mask] += s * state[basis]
                else:
                    # |1> -> -s|0> + c|1>
                    new_state[basis & ~mask] += -s * state[basis]
                    new_state[basis] += c * state[basis]
            state = new_state
        
        # Apply CZ gates between adjacent pairs (if not last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                mask_i = bit_masks[i]
                mask_j = bit_masks[i + 1]
                # Apply CZ: flip sign when both bits are 1
                for basis in range(dim):
                    if (basis & mask_i) and (basis & mask_j):
                        state[basis] *= -1
    
    return state

def _compute_energy_vectorized(state, n, coupling, field):
    """Compute <H> where H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 1 << n
    energy = 0.0
    
    # ZZ terms: diagonal in computational basis
    for i in range(n - 1):
        mask_i = 1 << (n - 1 - i)
        mask_j = 1 << (n - 1 - (i + 1))
        for basis in range(dim):
            z_i = 1.0 if (basis & mask_i) == 0 else -1.0
            z_j = 1.0 if (basis & mask_j) == 0 else -1.0
            prob = np.abs(state[basis]) ** 2
            energy += -coupling * z_i * z_j * prob
    
    # X terms: off-diagonal
    for i in range(n):
        mask = 1 << (n - 1 - i)
        for basis in range(dim):
            basis_flipped = basis ^ mask
            if basis < basis_flipped:
                cross = state[basis_flipped] * np.conj(state[basis])
                energy += -field * 2.0 * cross.real
    
    return energy

def _objective(params, n, coupling, field, layers):
    """Objective function for optimization."""
    theta = params.reshape(layers, n)
    state = _build_state_vectorized(theta, n, layers)
    energy = _compute_energy_vectorized(state, n, coupling, field)
    return energy

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=float)
    
    # ZZ terms
    for i in range(n - 1):
        mask_i = 1 << (n - 1 - i)
        mask_j = 1 << (n - 1 - (i + 1))
        for basis in range(dim):
            z_i = 1 if (basis & mask_i) == 0 else -1
            z_j = 1 if (basis & mask_j) == 0 else -1
            H[basis, basis] += -coupling * z_i * z_j
    
    # X terms
    for i in range(n):
        mask = 1 << (n - 1 - i)
        for basis in range(dim):
            basis_flipped = basis ^ mask
            H[basis, basis_flipped] += -field
    
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def _physics_init(n, field, layers):
    """Physics-based initialization based on expected ground state behavior."""
    theta = np.zeros((layers, n))
    
    # For the transverse field Ising model:
    # - Small field: ferromagnetic phase, angles near 0
    # - Large field: paramagnetic phase, angles near pi/2
    # The critical field for n-spin open chain is around 1.0
    
    # Use a smooth interpolation
    frac = np.clip(field / 1.5, 0, 1)
    base_angle = frac * np.pi / 2
    
    # Add some variation across layers for better optimization landscape
    theta[0, :] = base_angle
    theta[1, :] = base_angle * 0.8
    theta[2, :] = base_angle * 0.6
    
    return theta

def angles(n, coupling, field, layers):
    """Optimize circuit angles to approximate Ising ground state."""
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Physics-based initialization
    theta0 = _physics_init(n, field, layers)
    params0 = theta0.flatten()
    
    # Compute initial energy
    state0 = _build_state_vectorized(theta0, n, layers)
    energy0 = _compute_energy_vectorized(state0, n, coupling, field)
    
    best_params = params0.copy()
    best_energy = energy0
    
    # Try optimization with multiple strategies
    try:
        from scipy.optimize import minimize, differential_evolution
        
        # Strategy 1: Nelder-Mead from physics init
        result = minimize(_objective, params0, args=(n, coupling, field, layers),
                         method='Nelder-Mead', 
                         options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x.copy()
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            theta_best = best_params.reshape(layers, n)
            return theta_best
        
        # Strategy 2: Multiple restarts with Nelder-Mead
        n_restarts = 5
        for r in range(n_restarts):
            if time.time() - start_time > deadline - 20:
                break
            
            # Different random seeds for diversity
            np.random.seed(r * 7919 + n * 13 + int(field * 1000))
            
            # Mix of perturbations around best and random
            if r % 2 == 0:
                # Perturb around best
                perturbation = np.random.normal(0, 0.5, size=params0.shape)
                params_r = best_params + perturbation
            else:
                # Random initialization
                params_r = np.random.uniform(-np.pi/2, np.pi/2, size=params0.shape)
            
            result_r = minimize(_objective, params_r, args=(n, coupling, field, layers),
                               method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if result_r.fun < best_energy:
                best_energy = result_r.fun
                best_params = result_r.x.copy()
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            theta_best = best_params.reshape(layers, n)
            return theta_best
        
        # Strategy 3: Differential evolution for global search
        bounds = [(-np.pi, np.pi)] * (layers * n)
        result_de = differential_evolution(
            _objective, bounds, args=(n, coupling, field, layers),
            maxiter=100, popsize=15, tol=1e-8, seed=42,
            polish=True, init='sobol'
        )
        
        if result_de.fun < best_energy:
            best_energy = result_de.fun
            best_params = result_de.x.copy()
            
    except Exception as e:
        pass
    
    theta_best = best_params.reshape(layers, n)
    return theta_best
# EVOLVE-BLOCK-END
