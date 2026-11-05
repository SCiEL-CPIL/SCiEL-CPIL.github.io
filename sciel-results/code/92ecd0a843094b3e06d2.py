import numpy as np
import time
from scipy.optimize import minimize

# Global deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 135.0  # Safety margin before 150s kill

def _remaining():
    return _DEADLINE - time.time()

def _precompute(n, edges):
    """
    Precompute:
    - cut_vals: array of length 2^n with the number of cut edges for each state
    - edge_cuts: list of arrays, one per edge, indicating which states cut that edge
    - states: array of all state indices
    """
    dim = 1 << n
    states = np.arange(dim, dtype=np.int32)
    cut_vals = np.zeros(dim, dtype=np.float64)
    edge_cuts = []
    for (i, j) in edges:
        mask_i = 1 << i
        mask_j = 1 << j
        bits_i = (states & mask_i) != 0
        bits_j = (states & mask_j) != 0
        ec = (bits_i ^ bits_j).astype(np.float64)
        edge_cuts.append(ec)
        cut_vals += ec
    return cut_vals, edge_cuts, states

def _exact_maxcut(n, edges):
    """Compute exact MaxCut via brute force."""
    m = len(edges)
    if m == 0:
        return 0.0
    dim = 1 << n
    states = np.arange(dim, dtype=np.int32)
    best = 0
    for (i, j) in edges:
        mask_i = 1 << i
        mask_j = 1 << j
        bits_i = (states & mask_i) != 0
        bits_j = (states & mask_j) != 0
        ec = (bits_i ^ bits_j).astype(np.int32)
        for k in range(dim):
            pass  # We'll compute total cut per state efficiently below
    
    # More efficient: compute cut for all states at once
    cut_vals = np.zeros(dim, dtype=np.int32)
    for (i, j) in edges:
        mask_i = 1 << i
        mask_j = 1 << j
        bits_i = (states & mask_i) != 0
        bits_j = (states & mask_j) != 0
        cut_vals += (bits_i ^ bits_j).astype(np.int32)
    
    return float(np.max(cut_vals))

def _simulate_qaoa_fast(n, params, cut_vals, edge_cuts, states):
    """
    Fast QAOA simulation using precomputed data.
    params = [gamma1, beta1, gamma2, beta2]
    """
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute mixer unitary application via eigen-decomposition of B = sum X_i
    # B is the same for all graphs with same n, so we cache it
    
    # For now, use a simpler approach: apply cost phases and mixer via matrix
    # Actually, let's use the eigendecomposition approach for speed
    
    # Get or compute B eigendecomposition
    op_data = _get_operator_data(n)
    v = op_data['v']
    w = op_data['w']
    
    g1, b1, g2, b2 = params
    
    # Step 1: exp(-i gamma1 C)
    # C is diagonal in Z basis with eigenvalues cut_vals
    phase1 = np.exp(-1j * g1 * cut_vals)
    state = state * phase1
    
    # Step 2: exp(-i beta1 B)
    c = v.conj().T @ state
    c = c * np.exp(-1j * b1 * w)
    state = v @ c
    
    # Step 3: exp(-i gamma2 C)
    phase2 = np.exp(-1j * g2 * cut_vals)
    state = state * phase2
    
    # Step 4: exp(-i beta2 B)
    c = v.conj().T @ state
    c = c * np.exp(-1j * b2 * w)
    state = v @ c
    
    # Expected cut
    probs = np.abs(state)**2
    exp_cut = np.sum(probs * cut_vals)
    
    return exp_cut

# Cache for operator data
_operator_cache = {}

def _get_operator_data(n):
    if n in _operator_cache:
        return _operator_cache[n]
    
    dim = 1 << n
    B = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        for k in range(dim):
            k_flipped = k ^ (1 << i)
            B[k_flipped, k] += 1.0
    
    w, v = np.linalg.eigh(B)
    
    data = {'n': n, 'w': w, 'v': v}
    _operator_cache[n] = data
    return data

def _objective(params, n, cut_vals, edge_cuts, states, opt_cut):
    """Objective to minimize: negative of the ratio."""
    if _remaining() < 2.0:
        return 1.0  # Large penalty if out of time
    
    try:
        exp_cut = _simulate_qaoa_fast(n, params, cut_vals, edge_cuts, states)
        if opt_cut == 0:
            return 0.0
        return -exp_cut / opt_cut
    except Exception:
        return 1.0

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    Returns array of shape (p, 2) with rows (gamma, beta) in radians.
    """
    if p != 2:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Check time budget
    if _remaining() < 10.0:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    # Precompute graph data
    cut_vals, edge_cuts, states = _precompute(n, edges)
    opt_cut = float(np.max(cut_vals))
    
    if opt_cut == 0:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    # Initial guesses based on typical QAOA values
    init_points = [
        np.array([0.35, 0.25, 0.35, 0.25]),
        np.array([0.5, 0.3, 0.4, 0.2]),
        np.array([0.2, 0.1, 0.2, 0.1]),
        np.array([0.7, 0.5, 0.6, 0.4]),
        np.array([0.1, 0.4, 0.2, 0.3]),
        np.array([0.4, 0.1, 0.1, 0.4]),
        np.array([0.6, 0.2, 0.5, 0.3]),
    ]
    
    best_params = init_points[0].copy()
    best_val = _objective(best_params, n, cut_vals, edge_cuts, states, opt_cut)
    
    # Try multiple initial points with Nelder-Mead
    for init in init_points:
        if _remaining() < 5.0:
            break
        try:
            res = minimize(_objective, init, args=(n, cut_vals, edge_cuts, states, opt_cut),
                          method='Nelder-Mead',
                          options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # If time permits, try a second round with more iterations from best point
    if _remaining() > 15.0:
        try:
            res2 = minimize(_objective, best_params, args=(n, cut_vals, edge_cuts, states, opt_cut),
                          method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-7})
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Return shape (p, 2)
    # Rows: (gamma, beta)
    result = np.array([
        [best_params[0], best_params[1]],
        [best_params[2], best_params[3]]
    ])
    
    return result
