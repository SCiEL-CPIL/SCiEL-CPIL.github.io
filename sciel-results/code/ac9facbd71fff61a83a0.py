import numpy as np
import time
from functools import lru_cache

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 110.0  # safety margin before 120s

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _apply_cost(state, gamma, cost_diag):
    """Apply exp(-i gamma C) as diagonal multiplication."""
    return cost_diag * state

def _apply_mixing(state, beta, n):
    """Apply exp(-i beta * sum X) = tensor product of single-qubit rotations."""
    # M = [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
    c = np.cos(beta)
    s = np.sin(beta)
    # state shape: (2,)*n
    for k in range(n):
        # Move axis k to front, apply 2x2 matrix, move back
        state = np.moveaxis(state, k, 0)
        state = state.reshape(2, -1)
        # Apply M: new[0] = c*old[0] - 1j*s*old[1], new[1] = -1j*s*old[0] + c*old[1]
        new_state = np.empty_like(state)
        new_state[0] = c * state[0] - 1j * s * state[1]
        new_state[1] = -1j * s * state[0] + c * state[1]
        state = new_state
        state = state.reshape((2,) * n)
        state = np.moveaxis(state, 0, k)
    return state

def _expected_cut(n, edges, gammas, betas):
    """Compute expected cut value for given QAOA angles."""
    num_states = 1 << n
    # Initial state: |+>^n
    state = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
    
    # Precompute cost diagonal for each gamma
    # C = sum_{(i,j) in edges} (I - Z_i Z_j) / 2
    # exp(-i gamma C) = product over edges of exp(-i gamma (I - Z_i Z_j)/2)
    # For a basis state |x>, the eigenvalue of (I - Z_i Z_j)/2 is 0 if x_i == x_j, 1 if x_i != x_j
    # So exp(-i gamma * eigenvalue) = 1 if x_i == x_j, exp(-i gamma) if x_i != x_j
    
    for gamma, beta in zip(gammas, betas):
        # Apply cost
        if gamma != 0:
            # Compute diagonal: for each basis state, product over edges of exp(-i gamma) if cut
            diag = np.ones(num_states, dtype=complex)
            for (i, j) in edges:
                # For each basis state, check if bit i != bit j
                # bit i is (state >> i) & 1
                # Create mask: states where bit i != bit j
                # bit_i = (np.arange(num_states) >> i) & 1
                # bit_j = (np.arange(np.arange(num_states) >> j) & 1
                # cut_mask = bit_i != bit_j
                # diag[cut_mask] *= exp(-i gamma)
                indices = np.arange(num_states)
                bit_i = (indices >> i) & 1
                bit_j = (indices >> j) & 1
                cut_mask = (bit_i != bit_j)
                diag[cut_mask] *= np.exp(-1j * gamma)
            state = diag * state
        
        # Apply mixing
        if beta != 0:
            state = _apply_mixing(state, beta, n)
    
    # Expected cut = <psi|C|psi> = sum_{(i,j) in edges} P(i,j cut)
    # P(i,j cut) = sum_{x: x_i != x_j} |psi_x|^2
    probs = np.abs(state) ** 2
    expected = 0.0
    for (i, j) in edges:
        indices = np.arange(num_states)
        bit_i = (indices >> i) & 1
        bit_j = (indices >> j) & 1
        cut_mask = (bit_i != bit_j)
        expected += np.sum(probs[cut_mask])
    return expected

def _exact_maxcut(n, edges):
    """Compute exact MaxCut value by brute force."""
    num_states = 1 << n
    best = 0
    for x in range(num_states):
        cut = 0
        for (i, j) in edges:
            if ((x >> i) & 1) != ((x >> j) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best

def _optimize_angles(n, edges, p, deadline):
    """Optimize QAOA angles for given graph."""
    C_opt = _exact_maxcut(n, edges)
    if C_opt == 0:
        return np.zeros((p, 2))
    
    def objective(params):
        """Negative expected cut (to minimize)."""
        gammas = params[:p]
        betas = params[p:]
        try:
            ec = _expected_cut(n, edges, gammas, betas)
            return -ec
        except:
            return 1e10
    
    # Try multiple starting points
    best_params = None
    best_val = -np.inf
    
    # Grid of starting points
    start_points = []
    # Heuristic starting points
    start_points.append([0.5, 0.5] * 1)  # p=2: gamma1, gamma2, beta1, beta2
    start_points.append([0.35, 0.25] * 1)
    start_points.append([1.0, 0.5] * 1)
    start_points.append([0.8, 0.3] * 1)
    start_points.append([1.2, 0.4] * 1)
    start_points.append([0.6, 0.6] * 1)
    start_points.append([1.5, 0.2] * 1)
    start_points.append([0.4, 0.8] * 1)
    
    # Add random starts
    rng = np.random.RandomState(42)
    for _ in range(4):
        start_points.append(rng.uniform(0, np.pi, 4).tolist())
    
    from scipy.optimize import minimize
    
    for sp in start_points:
        if _time_remaining() < 15:
            break
        try:
            result = minimize(objective, sp, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except:
            pass
    
    if best_params is None:
        # Fallback
        best_params = np.array([0.5, 0.5, 0.5, 0.5])
    
    gammas = best_params[:p]
    betas = best_params[p:]
    return np.column_stack([gammas, betas])

# Cache
_angle_cache = {}

def angles(n, edges, p):
    # Normalize edges
    edges_list = tuple(sorted((min(i,j), max(i,j)) for i, j in edges))
    cache_key = (n, edges_list, p)
    
    if cache_key in _angle_cache:
        return _angle_cache[cache_key]
    
    if _time_remaining() < 5:
        # Not enough time, return default
        result = np.tile([0.35, 0.25], (p, 1))
        _angle_cache[cache_key] = result
        return result
    
    try:
        result = _optimize_angles(n, edges, p, _time_remaining())
    except Exception:
        result = np.tile([0.35, 0.25], (p, 1))
    
    _angle_cache[cache_key] = result
    return result
