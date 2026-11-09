"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()

def _precompute_graph(n, edges):
    """Precompute cut values and edge masks for vectorized operations."""
    dim = 1 << n
    states = np.arange(dim, dtype=np.int64)
    
    # Compute cut values vectorized
    cut_values = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        bits_i = (states >> i) & 1
        bits_j = (states >> j) & 1
        cut_values += (bits_i != bits_j).astype(np.float64)
    
    # Precompute edge masks for expected cut computation
    edge_masks = []
    for (i, j) in edges:
        mask_i = 1 << i
        mask_j = 1 << j
        # Create boolean array: True where bit_i != bit_j
        bits_i = (states >> i) & 1
        bits_j = (states >> j) & 1
        edge_masks.append(bits_i != bits_j)
    
    return cut_values, edge_masks

def _qaoa_expected_cut(n, edges, gammas, betas, precomputed=None):
    """
    Compute the expected cut size for a given QAOA circuit with p layers.
    Uses vectorized operations for speed.
    """
    if n > 12:
        return 0.5 * len(edges)
    
    dim = 1 << n
    
    if precomputed is None:
        cut_values, edge_masks = _precompute_graph(n, edges)
    else:
        cut_values, edge_masks = precomputed
    
    # Initial state: |+>^n = (1/sqrt(2^n)) * ones(2^n)
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in zip(gammas, betas):
        # Apply cost unitary: state[z] *= exp(-i gamma * cut_values[z])
        state = state * np.exp(-1j * gamma * cut_values)
        
        # Apply mixer: exp(-i beta * sum X_i) = prod_i exp(-i beta X_i)
        # Vectorized: for each qubit, reshape and apply 2x2 rotation
        cb = np.cos(beta)
        sb = np.sin(beta)
        
        for i in range(n):
            mask = 1 << i
            # Reshape to group pairs that differ in bit i
            # The states can be reshaped into blocks of size 2^(i+1) with stride 2^i
            # More simply: use reshape to (-1, 2) where each row is a pair
            # For bit i, states are grouped as: ... 0...0, 0...1, 1...0, 1...1
            # We need to reshape so that pairs (z, z^mask) are adjacent
            
            # Method: reshape into (dim // 2, 2) where each row is a pair
            # But this only works if we order states appropriately.
            # Alternative: use fancy indexing
            idx_even = np.arange(dim)
            # For each state z, its partner is z ^ mask
            # We want to apply the rotation to each pair
            # Let's reshape: group by (z >> (i+1)) and (z & (mask - 1))
            
            # Simpler approach using reshape:
            # If we reshape state into shape (dim >> (i+1), 2, 2^i), 
            # then axis 1 gives the bit i value
            shape = (dim >> (i + 1), 2, 1 << i)
            state_reshaped = state.reshape(shape)
            
            # Apply rotation: [a; b] -> [cb*a - i*sb*b; -i*sb*a + cb*b]
            a = state_reshaped[:, 0, :]  # bit i = 0
            b = state_reshaped[:, 1, :]  # bit i = 1
            
            new_a = cb * a - 1j * sb * b
            new_b = -1j * sb * a + cb * b
            
            state_reshaped[:, 0, :] = new_a
            state_reshaped[:, 1, :] = new_b
            # state is updated in place via the view
    
    # Compute expected cut vectorized
    probs = np.abs(state) ** 2
    expected_cut = 0.0
    for mask in edge_masks:
        expected_cut += np.sum(probs[mask])
    
    return expected_cut

def _exhaustive_maxcut(n, edges):
    """Compute the exact maximum cut by brute force for small n."""
    if n > 15:
        return 0.5 * len(edges)
    
    dim = 1 << n
    states = np.arange(dim, dtype=np.int64)
    best = 0
    for z in states:
        cut = 0
        for (i, j) in edges:
            bit_i = (z >> i) & 1
            bit_j = (z >> j) & 1
            if bit_i != bit_j:
                cut += 1
        if cut > best:
            best = cut
    
    return best

def _optimize_angles(n, edges, p, time_limit):
    """
    Optimize QAOA angles using DE for global search + Nelder-Mead for local polish.
    """
    from scipy.optimize import minimize, differential_evolution
    
    # Precompute graph data
    precomputed = _precompute_graph(n, edges)
    
    def objective(x):
        gammas = x[0::2]
        betas = x[1::2]
        return -_qaoa_expected_cut(n, edges, gammas, betas, precomputed)
    
    # Bounds: gamma in [0, pi], beta in [0, pi/2] typically, but let's be generous
    bounds = [(0, np.pi)] * p + [(0, np.pi / 2)] * p
    
    best_x = np.array([0.35, 0.25] * p, dtype=np.float64)
    best_val = objective(best_x)
    
    # Phase 1: Differential Evolution for global exploration
    if _time_left() > 20:
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=True,
                init='sobol',
                updating='deferred'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_x = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local polishing with Nelder-Mead from multiple starts
    np.random.seed(42)
    n_starts = 8
    for start_idx in range(n_starts):
        if _time_left() < 5:
            break
        
        if start_idx == 0:
            x_init = best_x.copy()
        elif start_idx == 1:
            x_init = np.array([0.35, 0.25] * p, dtype=np.float64)
        else:
            x_init = best_x + np.random.randn(2 * p) * 0.15
            # Clamp to reasonable bounds
            x_init = np.clip(x_init, [0, 0] * p, [np.pi, np.pi/2] * p)
        
        try:
            result = minimize(
                objective,
                x_init,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if _time_left() > 3:
        try:
            result = minimize(
                objective,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x

def angles(n, edges, p):
    """
    Return QAOA angles for a given graph.
    
    Args:
        n: number of vertices
        edges: list of (i, j) edges
        p: number of QAOA layers
    
    Returns:
        np.array of shape (p, 2), rows are (gamma, beta) in radians
    """
    # Check time
    if _time_left() < 5:
        return np.tile([0.35, 0.25], (p, 1))
    
    try:
        x_opt = _optimize_angles(n, edges, p, _time_left() - 5)
        gammas = x_opt[0::2]
        betas = x_opt[1::2]
        result = np.column_stack([gammas, betas])
        return result
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
