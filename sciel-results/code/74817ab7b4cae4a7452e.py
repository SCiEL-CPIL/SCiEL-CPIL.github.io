"""QAOA parameter design: multi-start optimization with physics-informed priors."""
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 118.0


def _time_left():
    return _DEADLINE - time.time()


def _build_cost_diag(n, edges):
    """Compute diagonal of MaxCut cost operator C."""
    num_states = 1 << n
    c_diag = np.zeros(num_states, dtype=np.float64)
    edge_array = np.array(edges, dtype=np.int32)
    if len(edge_array) == 0:
        return c_diag
    for state in range(num_states):
        bits = np.array([(state >> u) & 1 for u in range(n)], dtype=np.int32)
        # For each edge, check if endpoints differ
        diffs = bits[edge_array[:, 0]] != bits[edge_array[:, 1]]
        c_diag[state] = diffs.sum()
    return c_diag


def _expected_cut(params, n, c_diag):
    """Compute expected cut for QAOA parameters [g1, b1, g2, b2]."""
    dim = 1 << n
    state = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    
    g1, b1, g2, b2 = params
    
    # Precompute mixer rotation
    c1 = np.cos(b1)
    s1 = np.sin(b1)
    c2 = np.cos(b2)
    s2 = np.sin(b2)
    
    # Precompute cost phases
    phase_g1 = np.exp(-1j * g1 * c_diag)
    phase_g2 = np.exp(-1j * g2 * c_diag)
    
    # Apply cost layer 1
    state *= phase_g1
    
    # Apply mixer layer 1: tensor product of Rx(b1)
    # Rx(b) = [[cos(b), -i*sin(b)], [-i*sin(b), cos(b)]]
    # For each qubit axis, apply the 2x2 rotation
    for axis in range(n):
        state = np.moveaxis(state, axis, 0)
        s0 = state[0].copy()
        s1_arr = state[1].copy()
        new0 = c1 * s0 - 1j * s1 * s1_arr
        new1 = -1j * s1 * s0 + c1 * s1_arr
        state = np.stack([new0, new1], axis=0)
        state = np.moveaxis(state, 0, axis)
    
    # Apply cost layer 2
    state *= phase_g2
    
    # Apply mixer layer 2
    for axis in range(n):
        state = np.moveaxis(state, axis, 0)
        s0 = state[0].copy()
        s1_arr = state[1].copy()
        new0 = c2 * s0 - 1j * s2 * s1_arr
        new1 = -1j * s2 * s0 + c2 * s1_arr
        state = np.stack([new0, new1], axis=0)
        state = np.moveaxis(state, 0, axis)
    
    probs = np.abs(state) ** 2
    return float(np.dot(probs, c_diag))


def _optimize_angles(n, edges):
    """
    Multi-start optimization of QAOA angles.
    Returns array of shape (2, 2) with [gamma, beta] pairs.
    """
    c_diag = _build_cost_diag(n, edges)
    
    # Physics-informed priors for MaxCut QAOA (asymptotic values)
    # These are known good starting points
    priors = [
        np.array([0.37, 0.42, 0.37, 0.42]),
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([0.3, 0.5, 0.3, 0.5]),
        np.array([0.4, 0.4, 0.4, 0.4]),
        np.array([0.25, 0.6, 0.25, 0.6]),
        np.array([0.5, 0.3, 0.5, 0.3]),
        np.array([0.6, 0.4, 0.6, 0.4]),
        np.array([0.45, 0.45, 0.45, 0.45]),
    ]
    
    # Add some random points
    rng = np.random.default_rng(42)
    for _ in range(4):
        priors.append(rng.uniform(0.1, np.pi - 0.1, size=4))
    
    best_val = -np.inf
    best_params = np.array([0.5, 0.5, 0.5, 0.5])
    
    # Evaluate all initial points
    for init in priors:
        if _time_left() < 3.0:
            break
        val = _expected_cut(init, n, c_diag)
        if val > best_val:
            best_val = val
            best_params = init.copy()
    
    # Local refinement from the best initial point using Nelder-Mead
    if _time_left() > 8.0:
        def neg_cut(params):
            return -_expected_cut(params, n, c_diag)
        
        try:
            res = minimize(neg_cut, best_params.copy(), method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < neg_cut(best_params):
                best_params = res.x
                best_val = -res.fun
        except Exception:
            pass
    
    # Try local refinement from second-best initial points too
    # Sort initial points by their values
    init_vals = []
    for init in priors:
        val = _expected_cut(init, n, c_diag)
        init_vals.append((val, init.copy()))
    init_vals.sort(key=lambda x: -x[0])
    
    # Try refining the top few distinct initial points
    tried = [tuple(best_params)]
    for val, init in init_vals[:4]:
        if _time_left() < 5.0:
            break
        if tuple(np.round(init, 6)) in tried:
            continue
        tried.append(tuple(np.round(init, 6)))
        
        def neg_cut2(params):
            return -_expected_cut(params, n, c_diag)
        
        try:
            res = minimize(neg_cut2, init.copy(), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if -res.fun > best_val:
                best_val = -res.fun
                best_params = res.x
        except Exception:
            pass
    
    # High-precision polish pass
    if _time_left() > 4.0:
        def neg_cut_polish(params):
            return -_expected_cut(params, n, c_diag)
        
        try:
            res2 = minimize(neg_cut_polish, best_params.copy(), method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res2.fun < neg_cut_polish(best_params):
                best_params = res2.x
        except Exception:
            pass
    
    # Final polish with even tighter tolerances
    if _time_left() > 2.0:
        def neg_cut_final(params):
            return -_expected_cut(params, n, c_diag)
        
        try:
            res3 = minimize(neg_cut_final, best_params.copy(), method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            if res3.fun < neg_cut_final(best_params):
                best_params = res3.x
        except Exception:
            pass
    
    # Ensure parameters are within reasonable bounds
    best_params = np.clip(best_params, 0.0, np.pi)
    
    return best_params.reshape(2, 2)


def angles(n, edges, p):
    """
    Return QAOA angles for the given graph.
    n: number of qubits
    edges: list of edges (u, v)
    p: number of layers (fixed to 2 in this task)
    """
    if _time_left() < 1.0:
        return np.tile([0.5, 0.5], (p, 1))
    
    try:
        angles_2 = _optimize_angles(n, edges)
        
        if p == 2:
            return angles_2
        elif p < 2:
            return angles_2[:p]
        else:
            reps = (p + 1) // 2
            return np.tile(angles_2, (reps, 1))[:p]
            
    except Exception:
        return np.tile([0.5, 0.5], (p, 1))
