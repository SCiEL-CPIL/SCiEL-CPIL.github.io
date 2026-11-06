"""QAOA angle design for fixed-depth p=2 MaxCut on small graphs."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill
_CACHE = {}
_RNG = np.random.RandomState(12345)


def _precompute_graph(n, edges):
    """Precompute cut counts and flip indices for all 2^n states."""
    num_states = 2 ** n
    cut_counts = np.zeros(num_states, dtype=np.float64)
    m = len(edges)
    if m == 0:
        flip_indices = [np.arange(num_states) for _ in range(n)]
        return cut_counts, 0, flip_indices
    idxs = np.arange(num_states, dtype=np.int64)
    for u, v in edges:
        mask = ((idxs >> u) ^ (idxs >> v)) & 1
        cut_counts += mask
    # Precompute flip index arrays
    flip_indices = []
    for j in range(n):
        flip_mask = 1 << j
        flip_indices.append(idxs ^ flip_mask)
    return cut_counts, m, flip_indices


def _expected_cut_fast(n, cut_counts, flip_indices, gamma1, beta1, gamma2, beta2):
    """Compute expected cut value via vectorized state-vector simulation."""
    num_states = 2 ** n
    psi = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    cos_b1 = np.cos(beta1)
    sin_b1 = np.sin(beta1)
    cos_b2 = np.cos(beta2)
    sin_b2 = np.sin(beta2)
    
    phase1 = np.exp(-1j * gamma1 * cut_counts)
    phase2 = np.exp(-1j * gamma2 * cut_counts)
    
    for layer in range(2):
        if layer == 0:
            psi = psi * phase1
            cb, sb = cos_b1, sin_b1
        else:
            psi = psi * phase2
            cb, sb = cos_b2, sin_b2
        
        for j in range(n):
            flipped = flip_indices[j]
            psi = cb * psi - 1j * sb * psi[flipped]
    
    probs = np.abs(psi) ** 2
    return np.sum(probs * cut_counts)


def _optimize_angles(n, edges):
    """Optimize QAOA angles for a given graph using multi-stage strategy."""
    key = (n, tuple(sorted(edges)))
    if key in _CACHE:
        return _CACHE[key]
    
    if time.time() > _DEADLINE:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    cut_counts, m, flip_indices = _precompute_graph(n, edges)
    opt = int(np.max(cut_counts))
    if opt == 0:
        result = np.array([[0.0, 0.0], [0.0, 0.0]])
        _CACHE[key] = result
        return result
    
    def neg_ratio(params):
        g1, b1, g2, b2 = params
        ec = _expected_cut_fast(n, cut_counts, flip_indices, g1, b1, g2, b2)
        return -ec / opt
    
    # Stage 1: Coarse grid search with reduced resolution to save time
    best_val = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Use 5 points per dimension for faster grid search (625 evaluations)
    grid_points = 5
    grid = np.linspace(0.0, np.pi, grid_points)
    
    for g1 in grid:
        for b1 in grid:
            for g2 in grid:
                for b2 in grid:
                    if time.time() > _DEADLINE - 45:
                        break
                    val = neg_ratio([g1, b1, g2, b2])
                    if val > best_val:
                        best_val = val
                        best_params = np.array([g1, b1, g2, b2])
                if time.time() > _DEADLINE - 45:
                    break
            if time.time() > _DEADLINE - 45:
                break
        if time.time() > _DEADLINE - 45:
            break
    
    # Stage 2: Local refinement using Nelder-Mead from best grid point
    if time.time() < _DEADLINE - 35:
        try:
            from scipy.optimize import minimize
            result = minimize(
                neg_ratio,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Stage 3: Multi-start local refinement from diverse starting points
    if time.time() < _DEADLINE - 20:
        try:
            from scipy.optimize import minimize
            # Structured starting points based on QAOA theory
            structured_starts = [
                np.array([0.5, 0.3, 0.5, 0.3]),
                np.array([1.0, 0.5, 1.0, 0.5]),
                np.array([0.2, 0.8, 0.2, 0.8]),
                np.array([0.7, 0.2, 0.7, 0.2]),
                np.array([0.3, 0.6, 0.3, 0.6]),
                np.array([1.2, 0.4, 0.8, 0.7]),
                np.array([0.4, 1.1, 1.3, 0.3]),
                np.array([0.9, 0.9, 0.9, 0.9]),
                np.array([0.1, 0.1, 0.1, 0.1]),
                np.array([1.5, 0.2, 0.5, 1.2]),
                np.array([0.6, 0.4, 0.7, 0.5]),
                np.array([1.1, 0.3, 0.9, 0.6]),
                np.array([0.4, 0.7, 1.0, 0.4]),
                np.array([0.8, 0.6, 0.5, 0.8]),
            ]
            
            # Random starting points
            random_starts = []
            for _ in range(12):
                rs = _RNG.uniform(0.05, 2.8, size=4)
                random_starts.append(rs)
            
            all_starts = structured_starts + random_starts
            
            for start in all_starts:
                if time.time() > _DEADLINE - 12:
                    break
                result = minimize(
                    neg_ratio,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x
        except Exception:
            pass
    
    # Stage 4: Final polish with tighter tolerance
    if time.time() < _DEADLINE - 5:
        try:
            from scipy.optimize import minimize
            result = minimize(
                neg_ratio,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Ensure parameters are within valid range
    best_params = np.clip(best_params, 0.0, np.pi)
    
    angles_arr = np.array([[best_params[0], best_params[1]],
                           [best_params[2], best_params[3]]])
    _CACHE[key] = angles_arr
    return angles_arr


def angles(n, edges, p):
    """Return optimized QAOA angles for the given graph."""
    if time.time() > _DEADLINE:
        return np.tile([0.35, 0.25], (p, 1))
    
    try:
        result = _optimize_angles(n, edges)
        return result
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
