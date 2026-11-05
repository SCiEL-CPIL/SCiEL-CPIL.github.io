import numpy as np
import time
from scipy.optimize import minimize

# Global deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 138.0  # Safety margin before 150s kill

def _remaining():
    return _DEADLINE - time.time()

def _precompute(n, edges):
    """
    Precompute cut_vals for all 2^n states.
    """
    dim = 1 << n
    states = np.arange(dim, dtype=np.int32)
    cut_vals = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        mask_i = 1 << i
        mask_j = 1 << j
        bits_i = (states & mask_i) != 0
        bits_j = (states & mask_j) != 0
        cut_vals += (bits_i ^ bits_j).astype(np.float64)
    return cut_vals

def _apply_mixer(state, beta, n):
    """
    Apply exp(-i beta * sum_i X_i) to state vector.
    Since X_i commute, this equals prod_i exp(-i beta X_i).
    Each exp(-i beta X_i) is a 2x2 rotation on qubit i.
    """
    dim = 1 << n
    c = np.cos(beta)
    s = np.sin(beta)
    
    # For each qubit i, apply the rotation to all pairs differing in bit i
    for i in range(n):
        mask = 1 << i
        # Process in blocks: for each combination of other bits,
        # apply 2x2 rotation to the pair (state without bit i, state with bit i)
        # Vectorized: reshape state to (-1, 2) along axis i
        shape = list(state.shape)
        new_shape = shape[:i] + [2] + shape[i+1:]
        # Actually, let's do it differently
        # For each index k where bit i is 0, the partner is k ^ mask
        # We need to apply [[c, -is], [-is, c]] to (state[k], state[k^mask])
        
        # Create index arrays
        # All indices where bit i is 0
        # This is: for k in range(dim) where (k & mask) == 0
        # Partner: k | mask
        
        # Vectorized approach:
        # indices where bit i is 0
        # We can use: range(0, dim, 2*mask) for blocks, then within each block, first mask elements have bit i = 0
        # Actually simpler:
        idx0 = np.array([k for k in range(dim) if (k & mask) == 0])
        idx1 = idx0 | mask
        
        a = state[idx0].copy()
        b = state[idx1].copy()
        
        state[idx0] = c * a - 1j * s * b
        state[idx1] = -1j * s * a + c * b
    
    return state

def _simulate_qaoa(n, params, cut_vals):
    """
    Simulate QAOA with p=2 layers.
    params = [gamma1, beta1, gamma2, beta2]
    Returns expected cut value.
    """
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    g1, b1, g2, b2 = params
    
    # Step 1: exp(-i gamma1 * C)  -- diagonal
    state *= np.exp(-1j * g1 * cut_vals)
    
    # Step 2: exp(-i beta1 * B)  -- product of single-qubit rotations
    state = _apply_mixer(state, b1, n)
    
    # Step 3: exp(-i gamma2 * C)  -- diagonal
    state *= np.exp(-1j * g2 * cut_vals)
    
    # Step 4: exp(-i beta2 * B)  -- product of single-qubit rotations
    state = _apply_mixer(state, b2, n)
    
    # Expected cut
    probs = np.abs(state)**2
    exp_cut = np.sum(probs * cut_vals)
    
    return float(np.real(exp_cut))

def _objective(params, n, cut_vals, opt_cut):
    """Objective to minimize: negative of the ratio."""
    if _remaining() < 2.0:
        return 1.0
    try:
        exp_cut = _simulate_qaoa(n, params, cut_vals)
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
    cut_vals = _precompute(n, edges)
    opt_cut = float(np.max(cut_vals))
    
    if opt_cut == 0:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    # Multi-start optimization
    init_points = [
        np.array([0.35, 0.25, 0.35, 0.25]),
        np.array([0.5, 0.3, 0.4, 0.2]),
        np.array([0.2, 0.1, 0.2, 0.1]),
        np.array([0.7, 0.5, 0.6, 0.4]),
        np.array([0.1, 0.4, 0.2, 0.3]),
        np.array([0.4, 0.1, 0.1, 0.4]),
        np.array([0.6, 0.2, 0.5, 0.3]),
        np.array([0.3, 0.3, 0.3, 0.3]),
        np.array([0.8, 0.4, 0.7, 0.3]),
        np.array([0.15, 0.15, 0.15, 0.15]),
        np.array([0.45, 0.35, 0.45, 0.35]),
        np.array([0.25, 0.2, 0.25, 0.2]),
    ]
    
    best_params = init_points[0].copy()
    best_val = _objective(best_params, n, cut_vals, opt_cut)
    
    # Try each initial point with L-BFGS-B
    for init in init_points:
        if _remaining() < 8.0:
            break
        try:
            res = minimize(_objective, init, args=(n, cut_vals, opt_cut),
                          method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-7})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead from best point
    if _remaining() > 15.0:
        try:
            res2 = minimize(_objective, best_params, args=(n, cut_vals, opt_cut),
                          method='Nelder-Mead',
                          options={'maxiter': 400, 'xatol': 1e-7, 'fatol': 1e-9})
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if _remaining() > 20.0:
        try:
            res3 = minimize(_objective, best_params, args=(n, cut_vals, opt_cut),
                          method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res3.fun < best_val:
                best_val = res3.fun
                best_params = res3.x.copy()
        except Exception:
            pass
    
    # Return shape (p, 2)
    result = np.array([
        [best_params[0], best_params[1]],
        [best_params[2], best_params[3]]
    ])
    
    return result
