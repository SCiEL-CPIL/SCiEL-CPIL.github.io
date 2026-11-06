import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

# Global cache for precomputed graph data
_graph_cache = {}

def _simulate_qaoa_fast(n, edges, params, cut_sizes, flip_indices):
    """
    Simulate QAOA for given graph and parameters.
    params: array of shape (p, 2) with (gamma, beta).
    Returns expected cut size.
    """
    p = len(params)
    N = 1 << n
    state = np.ones(N, dtype=complex) / np.sqrt(N)
    
    for gamma, beta in params:
        # Apply Cost
        state *= np.exp(-1j * gamma * cut_sizes)
        
        # Apply Mixer
        c = np.cos(beta)
        s = -1j * np.sin(beta)
        
        for i in range(n):
            # Use precomputed flip indices
            state = c * state + s * state[flip_indices[i]]
            
    probs = np.abs(state)**2
    return np.sum(probs * cut_sizes)

def _expected_cut_from_params(params_flat, n, cut_sizes, flip_indices):
    """
    Compute expected cut from flat parameter array [gamma1, beta1, gamma2, beta2].
    """
    p = len(params_flat) // 2
    params = params_flat.reshape(p, 2)
    return _simulate_qaoa_fast(n, None, params, cut_sizes, flip_indices)

def angles(n, edges, p):
    n = int(n)
    # Normalize edges to list of tuples to ensure hashability and consistent sorting
    if isinstance(edges, np.ndarray):
        edges = [tuple(e) for e in edges]
    else:
        edges = [tuple(e) for e in edges]

    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    # Get or compute graph data
    edge_key = (n, tuple(sorted(edges)))
    if edge_key in _graph_cache:
        data = _graph_cache[edge_key]
    else:
        # Compute cut sizes
        N = 1 << n
        cut_sizes = np.zeros(N, dtype=float)
        states = np.arange(N, dtype=np.int64)
        for u, v in edges:
            bits_u = (states >> u) & 1
            bits_v = (states >> v) & 1
            cut_sizes += (bits_u != bits_v).astype(float)
        
        max_cut = float(np.max(cut_sizes))
        flip_indices = [np.arange(N) ^ (1 << i) for i in range(n)]
        
        data = {
            'cut_sizes': cut_sizes,
            'flip_indices': flip_indices,
            'max_cut': max_cut,
            'n': n
        }
        _graph_cache[edge_key] = data
    
    cut_sizes = data['cut_sizes']
    flip_indices = data['flip_indices']
    max_cut = float(data['max_cut'])
    
    if max_cut == 0.0:
        return np.zeros((p, 2))
    
    # Define objective function
    def local_obj(params):
        exp_cut = _expected_cut_from_params(params, n, cut_sizes, flip_indices)
        return float(-exp_cut / max_cut)
    
    bounds = [(0, np.pi)] * (2 * p)
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Coarse grid search to find good regions
    # For p=2, we have 4 parameters: gamma1, beta1, gamma2, beta2
    if time.time() - start_time < deadline - 40.0:
        # Use a moderate grid: 8 points per parameter = 4096 evaluations
        # For n=10, each eval is ~0.01ms, so 4096 * 0.01ms = 40ms total
        grid_points = 8
        grid = np.meshgrid(
            np.linspace(0, np.pi, grid_points),
            np.linspace(0, np.pi, grid_points),
            np.linspace(0, np.pi, grid_points),
            np.linspace(0, np.pi, grid_points),
            indexing='ij'
        )
        grid_flat = np.column_stack([g.ravel() for g in grid])
        
        # Evaluate in batches
        batch_size = 512
        for i in range(0, len(grid_flat), batch_size):
            if time.time() - start_time > deadline - 35.0:
                break
            batch = grid_flat[i:i+batch_size]
            for x0 in batch:
                try:
                    score = float(-local_obj(x0))
                    if score > best_score:
                        best_score = score
                        best_params = x0.copy()
                except Exception:
                    continue
    
    # Phase 2: Local optimization from top grid points and physics-informed guesses
    initial_guesses = [
        [0.5, 0.5, 0.5, 0.5],
        [0.7, 0.4, 0.7, 0.4],
        [0.3, 0.3, 0.3, 0.3],
        [0.2, 0.6, 0.2, 0.6],
        [1.0, 0.5, 1.0, 0.5],
        [0.4, 0.2, 0.4, 0.2],
        [0.8, 0.3, 0.8, 0.3],
        [0.1, 0.1, 0.1, 0.1],
        [0.6, 0.5, 0.6, 0.5],
        [0.9, 0.3, 0.9, 0.3],
    ]
    
    if best_params is not None:
        initial_guesses.append(best_params)
    
    # Add nearby points to the best grid point
    if best_params is not None:
        for delta in [-0.2, -0.1, 0.1, 0.2]:
            for j in range(4):
                x0 = best_params.copy()
                x0[j] = np.clip(x0[j] + delta, 0, np.pi)
                initial_guesses.append(x0)
    
    for x0 in initial_guesses:
        if time.time() - start_time > deadline - 20.0:
            break
        try:
            res = minimize(
                local_obj, x0, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            score = float(-res.fun)
            if score > best_score:
                best_score = score
                best_params = res.x
        except Exception:
            continue
    
    # Phase 3: Differential Evolution if time permits
    if time.time() - start_time < deadline - 15.0:
        try:
            de_result = differential_evolution(
                local_obj, 
                bounds, 
                maxiter=50, 
                popsize=20,
                seed=42,
                polish=True,
                tol=1e-6
            )
            de_score = float(-de_result.fun)
            if de_score > best_score:
                best_score = de_score
                best_params = de_result.x
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if best_params is not None and time.time() - start_time < deadline - 5.0:
        try:
            final_res = minimize(
                local_obj, best_params, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            final_score = float(-final_res.fun)
            if final_score > best_score:
                best_score = final_score
                best_params = final_res.x
        except Exception:
            pass
    
    if best_params is None:
        return np.array([[0.5, 0.5], [0.5, 0.5]])
    
    return np.array(best_params).reshape(p, 2)
