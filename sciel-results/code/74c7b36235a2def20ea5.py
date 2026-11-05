# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian matrix efficiently."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=float)
    
    # ZZ terms: diagonal
    for state in range(dim):
        zz_energy = 0.0
        for i in range(n - 1):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_ip1 = 1.0 if bit_ip1 == 0 else -1.0
            zz_energy += z_i * z_ip1
        H[state, state] += -coupling * zz_energy
    
    # X terms: off-diagonal
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            if state < flipped:  # avoid double counting
                H[flipped, state] += -field
                H[state, flipped] += -field
    
    return H

def _apply_ry_fast(state, theta, qubit, n):
    """Apply Ry(theta) to qubit on a state vector using vectorized operations."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    if abs(c) < 1e-15:
        c = 0.0
    if abs(s) < 1e-15:
        s = 0.0
    
    bit_mask = 1 << qubit
    dim = len(state)
    
    # Vectorized: find indices where qubit bit is 0
    indices = np.arange(dim)
    mask0 = (indices & bit_mask) == 0
    idx0 = indices[mask0]
    idx1 = idx0 | bit_mask
    
    new_state = state.copy()
    new_state[idx0] = c * state[idx0] - s * state[idx1]
    new_state[idx1] = s * state[idx0] + c * state[idx1]
    
    return new_state

def _apply_cz_fast(state, q1, q2, n):
    """Apply CZ between qubits q1 and q2 using vectorized operations."""
    mask1 = 1 << q1
    mask2 = 1 << q2
    both_mask = mask1 | mask2
    dim = len(state)
    
    indices = np.arange(dim)
    mask = (indices & both_mask) == both_mask
    state[mask] = -state[mask]
    
    return state

def _circuit_state_fast(n, layers, thetas):
    """Apply the full circuit to |0>^n and return the state vector using fast operations."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            state = _apply_ry_fast(state, thetas[layer, q], q, n)
        # Apply CZ between adjacent pairs (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz_fast(state, q, q + 1, n)
    
    return state

def _energy_fast(state, H):
    """Compute <state|H|state>."""
    return float(state @ H @ state)

def _exact_ground_energy(H):
    """Compute exact ground energy."""
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0])

def _physics_init(n, coupling, field, layers):
    """Physics-inspired initialization based on mean-field estimation."""
    thetas = np.zeros((layers, n), dtype=float)
    
    # For transverse field Ising, optimal single-qubit angle relates to arctan(h/J)
    # For open chain, boundary qubits see different effective fields
    if coupling != 0:
        base_angle = np.arctan2(field, coupling)
    else:
        base_angle = np.pi / 2
    
    # Distribute angles across layers with a smooth profile
    # Layer 0: small rotation
    thetas[0, :] = base_angle * 0.25
    # Layer 1: medium rotation
    thetas[1, :] = base_angle * 0.5
    # Layer 2: larger rotation
    thetas[2, :] = base_angle * 0.8
    
    # Add slight boundary effects
    if n > 2:
        thetas[:, 0] *= 1.1
        thetas[:, -1] *= 1.1
    
    return thetas

def _random_init(n, layers, scale=0.5, seed=None):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-scale, scale, size=(layers, n))

def _objective(params, n, coupling, field, layers, H, ground_energy):
    """Objective: mean energy gap per spin."""
    thetas = params.reshape(layers, n)
    state = _circuit_state_fast(n, layers, thetas)
    e = _energy_fast(state, H)
    return (e - ground_energy) / n

def _param_shift_gradient(params, n, coupling, field, layers, H, ground_energy):
    """
    Compute exact gradient using parameter-shift rule.
    For Ry(theta) gates, d/dtheta <psi|H|psi> = (E(theta+pi/2) - E(theta-pi/2)) / 2
    """
    thetas = params.reshape(layers, n)
    state = _circuit_state_fast(n, layers, thetas)
    e = _energy_fast(state, H)
    obj = (e - ground_energy) / n
    
    grad = np.zeros_like(params)
    shift = np.pi / 2
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += shift
        params_minus[i] -= shift
        
        state_plus = _circuit_state_fast(n, layers, params_plus.reshape(layers, n))
        state_minus = _circuit_state_fast(n, layers, params_minus.reshape(layers, n))
        
        e_plus = _energy_fast(state_plus, H)
        e_minus = _energy_fast(state_minus, H)
        
        grad[i] = (e_plus - e_minus) / (2 * n)
    
    return obj, grad

def angles(n, coupling, field, layers):
    """Return optimized circuit parameters."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Build Hamiltonian and compute exact ground energy
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _exact_ground_energy(H)
    
    n_params = layers * n
    
    best_thetas = None
    best_obj = np.inf
    
    # Generate diverse initializations
    inits = []
    
    # Physics-inspired
    phys = _physics_init(n, coupling, field, layers)
    inits.append(phys)
    
    # Variations around physics init
    for seed in range(4):
        init = phys + np.random.RandomState(seed).normal(0, 0.2, size=(layers, n))
        inits.append(init)
    
    # Random inits with different scales
    for scale, seed in [(0.3, 100), (0.5, 200), (0.8, 300), (1.0, 400)]:
        inits.append(_random_init(n, layers, scale, seed))
    
    # All zeros and all pi/2
    inits.append(np.zeros((layers, n)))
    inits.append(np.full((layers, n), np.pi / 2))
    
    # Evaluate initializations and keep top ones
    init_objs = []
    for init in inits:
        obj = _objective(init.flatten(), n, coupling, field, layers, H, ground_energy)
        init_objs.append((obj, init))
        if obj < best_obj:
            best_obj = obj
            best_thetas = init.copy()
    
    # Sort by objective and take best few for optimization
    init_objs.sort(key=lambda x: x[0])
    
    # Optimize from best initializations using parameter-shift gradient
    n_optimize = min(5, len(init_objs))
    for idx in range(n_optimize):
        if time.time() > deadline - 30:
            break
        
        obj_val, init = init_objs[idx]
        
        try:
            # First pass: L-BFGS-B with parameter-shift gradient
            result = minimize(
                _objective,
                init.flatten(),
                jac=_param_shift_gradient,
                args=(n, coupling, field, layers, H, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_thetas = result.x.reshape(layers, n)
                
        except Exception:
            continue
    
    # High-precision polish on the best solution
    if time.time() < deadline - 20 and best_thetas is not None:
        try:
            result = minimize(
                _objective,
                best_thetas.flatten(),
                jac=_param_shift_gradient,
                args=(n, coupling, field, layers, H, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_thetas = result.x.reshape(layers, n)
        except Exception:
            pass
    
    # Final ultra-precision polish if time permits
    if time.time() < deadline - 10 and best_thetas is not None:
        try:
            result = minimize(
                _objective,
                best_thetas.flatten(),
                jac=_param_shift_gradient,
                args=(n, coupling, field, layers, H, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 200}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_thetas = result.x.reshape(layers, n)
        except Exception:
            pass
    
    if best_thetas is None:
        return _physics_init(n, coupling, field, layers)
    
    return best_thetas
# EVOLVE-BLOCK-END
