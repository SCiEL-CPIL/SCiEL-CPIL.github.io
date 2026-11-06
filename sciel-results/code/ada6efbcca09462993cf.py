"""Rewritten program: Symmetry-reduced grid search + local polish for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_start_time = time.time()
_TIME_LIMIT = 148.0  # Safety margin before 150s kill
_cache = {}

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    for idx in range(dim):
        # Diagonal: ZZ terms
        h_diag = 0.0
        for i in range(n - 1):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            h_diag += -coupling * zi * zip1
        H[idx, idx] = h_diag
        
        # Off-diagonal: X terms
        for i in range(n):
            other = idx ^ (1 << i)
            H[idx, other] += -field
    
    return H

def _exact_ground_energy(n, coupling, field):
    """Compute the exact ground state energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues = np.linalg.eigvalsh(H)
    return np.real(eigenvalues[0])

def _apply_ry_vec(state, theta, qubit, n):
    """Apply Ry(theta) gate to qubit using vectorized operations."""
    if abs(theta) < 1e-15:
        return state
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    
    shape = [2] * n
    state_r = state.reshape(shape)
    state_moved = np.moveaxis(state_r, qubit, 0)
    
    new_state_moved = np.empty_like(state_moved)
    new_state_moved[0] = c * state_moved[0] - s * state_moved[1]
    new_state_moved[1] = s * state_moved[0] + c * state_moved[1]
    
    state_moved_back = np.moveaxis(new_state_moved, 0, qubit)
    return state_moved_back.reshape(-1)

def _apply_cz_vec(state, q1, q2, n):
    """Apply CZ gate between qubits q1 and q2."""
    if q1 == q2:
        return state
    shape = [2] * n
    state_r = state.reshape(shape)
    state_moved = np.moveaxis(state_r, [q1, q2], [-2, -1])
    
    idx = [slice(None)] * (n - 2) + [1, 1]
    state_moved[idx] *= -1.0
    
    state_moved_back = np.moveaxis(state_moved, [-2, -1], [q1, q2])
    return state_moved_back.reshape(-1)

def _circuit_state_symmetric(unique_params, n, layers):
    """
    Compute the state produced by the circuit with reflection symmetry.
    unique_params has shape (layers, n_unique) where n_unique = (n+1)//2.
    The full thetas are symmetric: theta[layer, q] = theta[layer, n-1-q].
    """
    n_unique = (n + 1) // 2
    thetas = np.zeros((layers, n))
    for layer in range(layers):
        for q in range(n_unique):
            theta = unique_params[layer, q]
            thetas[layer, q] = theta
            thetas[layer, n - 1 - q] = theta
    
    state = np.zeros(2 ** n, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            state = _apply_ry_vec(state, thetas[layer, q], q, n)
        
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz_vec(state, q, q + 1, n)
    
    return state

def _energy_of_state(state, n, coupling, field):
    """Compute <state|H|state> for the Ising Hamiltonian."""
    dim = 2 ** n
    energy = 0.0
    
    # ZZ terms (diagonal)
    for i in range(n - 1):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            amp = state[idx]
            energy += -coupling * zi * zip1 * (amp.real ** 2 + amp.imag ** 2)
    
    # X terms (off-diagonal)
    for i in range(n):
        for idx in range(dim):
            other = idx ^ (1 << i)
            amp = state[idx]
            amp_other = state[other]
            # <psi|H_X|psi> contribution: -field * (psi_i * psi_other + psi_other * psi_i)
            # = -field * 2 * Re(conj(psi_i) * psi_other)
            cross = np.conj(amp) * amp_other
            energy += -field * 2.0 * np.real(cross)
    
    return energy

def _objective_energy_gap(unique_params, n, coupling, field, e0):
    """Objective: minimize (circuit_energy - exact_ground_energy) / n."""
    state = _circuit_state_symmetric(unique_params, n, layers=3)
    e_circuit = _energy_of_state(state, n, coupling, field)
    return (e_circuit - e0) / n

def _optimize_symmetry(n, coupling, field, layers):
    """
    Optimize using reflection symmetry: theta[layer, q] = theta[layer, n-1-q].
    Uses grid search over reduced space followed by local polish.
    """
    e0 = _exact_ground_energy(n, coupling, field)
    n_unique = (n + 1) // 2
    total_params = layers * n_unique
    
    def _obj_flat(flat_params):
        params_2d = flat_params.reshape(layers, n_unique)
        return _objective_energy_gap(params_2d, n, coupling, field, e0)
    
    # Phase 1: Coarse grid search
    # Grid points per dimension: start coarse, refine based on time
    grid_size = 7  # 7 points per unique angle
    
    grid_vals = np.linspace(-np.pi, np.pi, grid_size)
    
    # Generate all combinations
    grid_points = np.array(np.meshgrid(*([grid_vals] * total_params))).T.reshape(-1, total_params)
    
    best_val = np.inf
    best_params = None
    
    for i, point in enumerate(grid_points):
        if time.time() - _start_time > _TIME_LIMIT - 10:
            break
        try:
            val = _obj_flat(point)
            if val < best_val:
                best_val = val
                best_params = point.copy()
        except Exception:
            pass
    
    # Phase 2: Local polish from best grid point and nearby points
    polish_starts = []
    if best_params is not None:
        polish_starts.append(best_params)
    
    # Add some random starts for robustness
    rng = np.random.default_rng(42)
    for _ in range(20):
        polish_starts.append(rng.uniform(-np.pi, np.pi, size=total_params))
    
    # Also add heuristic starts
    ratio = field / coupling
    base_theta = np.arctan(ratio)
    polish_starts.append(np.full(total_params, base_theta))
    polish_starts.append(np.zeros(total_params))
    polish_starts.append(np.full(total_params, np.pi / 2))
    
    nm_options = {
        'maxiter': 200,
        'fatol': 1e-12,
        'xatol': 1e-10,
        'adaptive': True
    }
    
    for x0 in polish_starts:
        if time.time() - _start_time > _TIME_LIMIT - 5:
            break
        try:
            res = minimize(
                _obj_flat,
                x0,
                method='Nelder-Mead',
                options=nm_options
            )
            val = res.fun
            if val < best_val:
                best_val = val
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        best_params = np.zeros(total_params)
    
    # Expand best_params to full shape (layers, n)
    full_params = np.zeros((layers, n))
    for layer in range(layers):
        for q in range(n_unique):
            theta = best_params[layer * n_unique + q]
            full_params[layer, q] = theta
            full_params[layer, n - 1 - q] = theta
    
    return full_params

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta

    try:
        theta = _optimize_symmetry(n, coupling, field, layers)
        _cache[key] = theta
        return theta
    except Exception:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        _cache[key] = theta
        return theta
# EVOLVE-BLOCK-END
