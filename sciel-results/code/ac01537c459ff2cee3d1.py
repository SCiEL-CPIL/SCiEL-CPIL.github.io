"""QAOA angle design for fixed-depth p=2 MaxCut on small graphs."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 130.0  # safety margin before 150s kill
_CACHE = {}
_FLIP_CACHE = {}


def _precompute_graph(n, edges):
    """Precompute cut counts for all 2^n states."""
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


def _get_flip_indices(n):
    """Get precomputed flip index arrays for all qubits."""
    if n in _FLIP_CACHE:
        return _FLIP_CACHE[n]
    num_states = 2 ** n
    idxs = np.arange(num_states, dtype=np.int64)
    flips = [idxs ^ (1 << j) for j in range(n)]
    _FLIP_CACHE[n] = flips
    return flips


def _expected_cut_fast(n, cut_counts, gamma1, beta1, gamma2, beta2):
    """Compute expected cut value via fully vectorized state-vector simulation."""
    num_states = 2 ** n
    psi = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    cos_b1 = np.cos(beta1)
    sin_b1 = np.sin(beta1)
    cos_b2 = np.cos(beta2)
    sin_b2 = np.sin(beta2)
    
    phase1 = np.exp(-1j * gamma1 * cut_counts)
    phase2 = np.exp(-1j * gamma2 * cut_counts)
    
    flips = _get_flip_indices(n)
    
    for layer in range(2):
        if layer == 0:
            psi = psi * phase1
            cb, sb = cos_b1, sin_b1
        else:
            psi = psi * phase2
            cb, sb = cos_b2, sin_b2
        
        for j in range(n):
            flipped = flips[j]
            psi = cb * psi - 1j * sb * psi[flipped]
    
    probs = np.abs(psi) ** 2
    return np.sum(probs * cut_counts)


def _exact_optimum(n, edges, cut_counts):
    """Compute exact MaxCut optimum."""
    return int(np.max(cut_counts))


def _optimize_angles(n, edges):
    """Optimize QAOA angles for a given graph."""
    key = (n, tuple(sorted(edges)))
    if key in _CACHE:
        return _CACHE[key]
    
    if time.time() > _DEADLINE:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    cut_counts, m = _precompute_graph(n, edges)
    opt = _exact_optimum(n, edges, cut_counts)
    if opt == 0:
        result = np.array([[0.0, 0.0], [0.0, 0.0]])
        _CACHE[key] = result
        return result
    
    def neg_ratio(params):
        g1, b1, g2, b2 = params
        ec = _expected_cut_fast(n, cut_counts, g1, b1, g2, b2)
        return -ec / opt
    
    best_val = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Stage 1: Focused grid search in known-good QAOA region
    # For p=2 MaxCut, typical optimal angles are in [0.1, 1.5] for gamma and [0.1, 1.0] for beta
    g_grid = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5])
    b_grid = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0])
    
    for g1 in g_grid:
        for b1 in b_grid:
            for g2 in g_grid:
                for b2 in b_grid:
                    if time.time() > _DEADLINE - 20:
                        break
                    val = neg_ratio([g1, b1, g2, b2])
                    if val > best_val:
                        best_val = val
                        best_params = np.array([g1, b1, g2, b2])
    
    # Stage 2: Local refinement from best grid point
    if time.time() < _DEADLINE - 25:
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
    
    # Stage 3: Multi-start local refinement from diverse strategic points
    if time.time() < _DEADLINE - 10:
        try:
            from scipy.optimize import minimize
            # Strategic starting points based on known QAOA behavior for p=2 MaxCut
            # Include points with different gamma/beta relationships
            starts = [
                np.array([0.3, 0.2, 0.3, 0.2]),
                np.array([0.5, 0.3, 0.5, 0.3]),
                np.array([0.7, 0.4, 0.7, 0.4]),
                np.array([1.0, 0.5, 1.0, 0.5]),
                np.array([0.4, 0.5, 0.4, 0.5]),
                np.array([1.2, 0.3, 0.8, 0.6]),
                np.array([0.6, 0.8, 0.4, 0.4]),
                np.array([1.5, 0.2, 1.0, 0.7]),
                np.array([0.2, 0.6, 0.9, 0.3]),
                np.array([0.8, 0.2, 1.2, 0.5]),
                np.array([1.0, 0.8, 0.5, 0.4]),
                np.array([0.5, 0.5, 1.0, 0.3]),
            ]
            for start in starts:
                if time.time() > _DEADLINE - 5:
                    break
                result = minimize(
                    neg_ratio,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x
        except Exception:
            pass
    
    # Stage 4: Final high-precision polish on best candidate
    if time.time() < _DEADLINE - 3:
        try:
            from scipy.optimize import minimize
            result = minimize(
                neg_ratio,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Clamp to valid range [0, pi]
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
