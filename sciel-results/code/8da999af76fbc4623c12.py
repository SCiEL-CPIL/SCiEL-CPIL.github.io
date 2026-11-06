"""QAOA angle design for fixed-depth p=2 MaxCut on small graphs."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 130.0  # safety margin before 150s kill
_CACHE = {}


def _precompute_graph(n, edges):
    """Precompute cut counts for all 2^n states using vectorized operations."""
    num_states = 2 ** n
    cut_counts = np.zeros(num_states, dtype=np.float64)
    m = len(edges)
    if m == 0:
        return cut_counts, 0
    idxs = np.arange(num_states, dtype=np.int64)
    for u, v in edges:
        mask = ((idxs >> u) ^ (idxs >> v)) & 1
        cut_counts += mask
    return cut_counts, m


def _expected_cut_fast(n, cut_counts, gamma1, beta1, gamma2, beta2):
    """Compute expected cut value via vectorized state-vector simulation."""
    num_states = 2 ** n
    psi = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    cos_b1 = np.cos(beta1)
    sin_b1 = np.sin(beta1)
    cos_b2 = np.cos(beta2)
    sin_b2 = np.sin(beta2)
    
    phase1 = np.exp(-1j * gamma1 * cut_counts)
    phase2 = np.exp(-1j * gamma2 * cut_counts)
    
    idxs = np.arange(num_states, dtype=np.int64)
    
    for layer in range(2):
        if layer == 0:
            psi = psi * phase1
            cb, sb = cos_b1, sin_b1
        else:
            psi = psi * phase2
            cb, sb = cos_b2, sin_b2
        
        for j in range(n):
            flip_mask = 1 << j
            flipped = idxs ^ flip_mask
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
    
    cut_counts, m = _precompute_graph(n, edges)
    opt = int(np.max(cut_counts))
    if opt == 0:
        result = np.array([[0.0, 0.0], [0.0, 0.0]])
        _CACHE[key] = result
        return result
    
    def neg_ratio(params):
        g1, b1, g2, b2 = params
        ec = _expected_cut_fast(n, cut_counts, g1, b1, g2, b2)
        return -ec / opt
    
    # Stage 1: Coarse grid search with adaptive resolution
    best_val = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Use a moderate grid - 8 points per dimension
    grid_points = 8
    grid = np.linspace(0.0, np.pi, grid_points)
    
    count = 0
    for g1 in grid:
        for b1 in grid:
            for g2 in grid:
                for b2 in grid:
                    if time.time() > _DEADLINE - 30:
                        break
                    val = neg_ratio([g1, b1, g2, b2])
                    if val > best_val:
                        best_val = val
                        best_params = np.array([g1, b1, g2, b2])
                    count += 1
                if time.time() > _DEADLINE - 30:
                    break
            if time.time() > _DEADLINE - 30:
                break
        if time.time() > _DEADLINE - 30:
            break
    
    # Stage 2: Local refinement using Nelder-Mead from best grid point
    if time.time() < _DEADLINE - 20:
        try:
            from scipy.optimize import minimize
            result = minimize(
                neg_ratio,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Stage 3: Multi-start local refinement from diverse starting points
    if time.time() < _DEADLINE - 10:
        try:
            from scipy.optimize import minimize
            # Generate diverse starting points based on graph structure
            starts = [
                np.array([0.5, 0.3, 0.5, 0.3]),
                np.array([1.0, 0.5, 1.0, 0.5]),
                np.array([0.2, 0.8, 0.2, 0.8]),
                np.array([0.7, 0.2, 0.7, 0.2]),
                np.array([0.3, 0.6, 0.3, 0.6]),
            ]
            for start in starts:
                if time.time() > _DEADLINE - 5:
                    break
                result = minimize(
                    neg_ratio,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
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
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
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
