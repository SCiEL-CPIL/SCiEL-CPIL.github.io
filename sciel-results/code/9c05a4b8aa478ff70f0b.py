"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize as _scipy_min

_energy_cache = {}
_hamiltonian_cache = {}

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    key = (n, coupling, field)
    if key in _hamiltonian_cache:
        return _hamiltonian_cache[key]
    
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Transverse field term: -field * sum X_i
    for i in range(n):
        for state_idx in range(dim):
            if (state_idx >> i) & 1 == 0:
                partner = state_idx | (1 << i)
                H[state_idx, partner] += -field
                H[partner, state_idx] += -field
    
    # Coupling term: -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        for state_idx in range(dim):
            bi = (state_idx >> i) & 1
            bj = (state_idx >> (i + 1)) & 1
            zz = (1 - 2 * bi) * (1 - 2 * bj)
            H[state_idx, state_idx] += -coupling * zz
    
    _hamiltonian_cache[key] = H
    return H

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via dense diagonalization."""
    key = (n, coupling, field)
    if key in _energy_cache:
        return _energy_cache[key]
    
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    ground_energy = float(eigvals[0])
    _energy_cache[key] = ground_energy
    return ground_energy

def _apply_circuit(params, n, layers):
    """Apply the 3-layer Ry/CZ circuit to |0>^n using vectorized operations."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    # Precompute bit masks
    qubit_masks = [1 << q for q in range(n)]
    adjacent_pairs = [(q, q + 1) for q in range(n - 1)]
    
    for layer in range(layers):
        # Apply Ry to every qubit (vectorized)
        for q in range(n):
            mask = qubit_masks[q]
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            # Separate states by qubit value
            bit0_mask = np.zeros(dim, dtype=bool)
            bit0_mask[~((np.arange(dim) >> q) & 1)] = True
            bit1_mask = ~bit0_mask
            
            # For bit=0: new[0] = c*old[0], new[1] = s*old[0]
            # For bit=1: new[0] = -s*old[1], new[1] = c*old[1]
            old_0 = state[bit0_mask].copy()
            old_1 = state[bit1_mask].copy()
            
            # Indices where bit is 0
            idx0 = np.where(bit0_mask)[0]
            idx1 = idx0 | mask
            
            new_state = state.copy()
            new_state[idx0] = c * old_0
            new_state[idx1] = s * old_0
            new_state[idx1] += c * old_1
            new_state[idx0] += -s * old_1
            
            state = new_state
        
        # Apply CZ on adjacent pairs (except after final layer)
        if layer < layers - 1:
            for q1, q2 in adjacent_pairs:
                m1 = qubit_masks[q1]
                m2 = qubit_masks[q2]
                # Find states where both bits are 1
                both_one = np.zeros(dim, dtype=bool)
                both_one[((np.arange(dim) & m1) != 0) & ((np.arange(dim) & m2) != 0)] = True
                state[both_one] *= -1
    
    return state

def _circuit_energy(params, n, coupling, field, H):
    """Compute the energy expectation of the circuit state."""
    state = _apply_circuit(params, n, len(params))
    return float(np.real(np.conj(state) @ H @ state))

def _mean_field_init(n, coupling, field, layers):
    """Initialize parameters using mean-field / product-state approximation.
    
    For the transverse-field Ising model, a product state |+>^n with small
    corrections is a reasonable starting point. We use a simple heuristic:
    - Last layer: pi/2 (maximal superposition)
    - Earlier layers: small angles scaled by field strength
    """
    params = np.zeros((layers, n))
    
    # Last layer: strong superposition
    params[-1, :] = np.pi / 2.0
    
    # Earlier layers: small rotation, scaled by field
    field_scale = min(field / 1.65, 1.0)
    params[0, :] = 0.3 * field_scale
    params[1, :] = 0.5 * field_scale
    
    return params

def _objective(params, n, coupling, field, H, ground_energy):
    """Objective: (circuit_energy - ground_energy) / n."""
    energy = _circuit_energy(params, n, coupling, field, H)
    return (energy - ground_energy) / n

def _objective_grad(params, n, coupling, field, H, ground_energy):
    """Objective with finite-difference gradient."""
    energy = _circuit_energy(params, n, coupling, field, H)
    obj = (energy - ground_energy) / n
    
    eps = 1e-6
    grad = np.zeros_like(params)
    for idx in np.ndindex(params.shape):
        orig = params[idx]
        params[idx] = orig + eps
        e_plus = _circuit_energy(params, n, coupling, field, H)
        params[idx] = orig - eps
        e_minus = _circuit_energy(params, n, coupling, field, H)
        params[idx] = orig
        grad[idx] = (e_plus - e_minus) / (2.0 * eps) / n
    
    return obj, grad

def angles(n, coupling, field, layers):
    """Optimize circuit parameters to approximate ground state."""
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    # Compute exact ground energy and Hamiltonian
    ground_energy = _exact_ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    
    best_params = _mean_field_init(n, coupling, field, layers)
    best_obj = _objective(best_params.ravel(), n, coupling, field, H, ground_energy)
    
    # Phase 1: Nelder-Mead from mean-field init (robust, no gradient needed)
    try:
        result_nm = _scipy_min(
            lambda p: _objective(p, n, coupling, field, H, ground_energy),
            best_params.ravel().copy(),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7}
        )
        if result_nm.fun < best_obj:
            best_obj = result_nm.fun
            best_params = result_nm.x.reshape(layers, n).copy()
    except Exception:
        pass
    
    if time.time() > deadline - 30:
        return best_params
    
    # Phase 2: L-BFGS-B from best NM result + a few random restarts
    np.random.seed(42)
    
    # Start from NM result
    try:
        result_lbfgs = _scipy_min(
            _objective_grad,
            best_params.ravel().copy(),
            args=(n, coupling, field, H, ground_energy),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 150, 'gtol': 1e-7, 'maxls': 30}
        )
        if result_lbfgs.fun < best_obj:
            best_obj = result_lbfgs.fun
            best_params = result_lbfgs.x.reshape(layers, n).copy()
    except Exception:
        pass
    
    # Random restarts with L-BFGS-B
    num_restarts = 8
    for rs in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        params_init = np.random.uniform(-np.pi, np.pi, size=(layers, n))
        
        try:
            result = _scipy_min(
                _objective_grad,
                params_init.ravel().copy(),
                args=(n, coupling, field, H, ground_energy),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'gtol': 1e-6, 'maxls': 20}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.reshape(layers, n).copy()
        except Exception:
            continue
    
    return best_params
# EVOLVE-BLOCK-END
