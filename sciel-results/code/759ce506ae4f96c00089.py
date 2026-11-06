import numpy as np
import time
from scipy.optimize import minimize

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
    deadline = 130.0  # Safety margin from 150s limit
    
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
    
    # Phase 1: Dense Deterministic Grid Search
    # Since simulation is fast, we can afford a finer grid than DE might sample.
    # We use a 3-stage grid: coarse, medium, fine around best coarse point.
    
    def grid_search(grid_res, bounds_list, top_k=10):
        """
        Perform a grid search over the given resolution.
        Returns top_k candidates (params, score) sorted by score.
        """
        grid_points = []
        # Generate grid points
        ranges = [np.linspace(b[0], b[1], grid_res) for b in bounds_list]
        
        # Iterate through all combinations
        # For 4 dimensions, this is grid_res^4
        # If grid_res=10, 10^4 = 10000 evals. Very fast.
        import itertools
        for combo in itertools.product(*ranges):
            params = np.array(combo)
            score = -local_obj(params)
            grid_points.append((score, params))
            
        # Sort by score descending
        grid_points.sort(key=lambda x: x[0], reverse=True)
        return grid_points[:top_k]
    
    # Stage 1: Coarse Grid (e.g., 10 points per dim -> 10k evals)
    coarse_results = grid_search(10, bounds, top_k=20)
    if coarse_results:
        best_score = max(best_score, coarse_results[0][0])
        best_params = coarse_results[0][1]
        
    # Stage 2: Medium Grid around best coarse point
    # Create smaller bounds around the best coarse point
    step = (bounds[0][1] - bounds[0][0]) / 10.0
    med_bounds = []
    for i, b in enumerate(bounds):
        low = max(0, best_params[i] - 2 * step)
        high = min(np.pi, best_params[i] + 2 * step)
        med_bounds.append((low, high))
    
    med_results = grid_search(20, med_bounds, top_k=10)
    if med_results:
        if med_results[0][0] > best_score:
            best_score = med_results[0][0]
            best_params = med_results[0][1]
            
    # Stage 3: Fine Grid around best medium point
    step_med = (med_bounds[0][1] - med_bounds[0][0]) / 20.0
    fine_bounds = []
    for i, b in enumerate(med_bounds):
        low = max(0, best_params[i] - 1.5 * step_med)
        high = min(np.pi, best_params[i] + 1.5 * step_med)
        fine_bounds.append((low, high))
        
    fine_results = grid_search(30, fine_bounds, top_k=5)
    if fine_results:
        if fine_results[0][0] > best_score:
            best_score = fine_results[0][0]
            best_params = fine_results[0][1]
            
    # Phase 2: Local Optimization from Top Grid Candidates
    # Use L-BFGS-B or Powell for high precision local search
    # We take the top 3 from the fine grid and the best from medium grid
    candidates = [fine_results[0][1]] if fine_results else [best_params]
    if med_results and len(candidates) < 3:
        candidates.append(med_results[0][1])
    if coarse_results and len(candidates) < 3:
        candidates.append(coarse_results[0][1])
        
    # Ensure no duplicates in candidates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(np.round(c, 4))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    for x0 in unique_candidates:
        if time.time() - start_time > deadline - 10.0:
            break
        try:
            # L-BFGS-B is bounded and uses gradient estimates, often faster and more robust
            res = minimize(
                local_obj, x0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success and res.fun < best_score:
                # res.fun is negative score, so lower is better
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_params = res.x
        except Exception:
            continue
        
    # Phase 3: Final Polish with Powell (derivative-free but robust)
    if best_params is not None and time.time() - start_time < deadline - 2.0:
        try:
            final_res = minimize(
                local_obj, best_params, method='Powell',
                bounds=bounds,
                options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            if final_res.fun < best_score:
                current_score = -final_res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_params = final_res.x
        except Exception:
            pass

    if best_params is None:
        return np.array([[0.5, 0.5], [0.5, 0.5]])
    
    return np.array(best_params).reshape(p, 2)
