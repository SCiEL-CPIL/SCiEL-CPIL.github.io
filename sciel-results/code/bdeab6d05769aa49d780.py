"""QAOA parameter design for fixed-depth (p=2) MaxCut."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_DEADLINE = 130.0  # seconds, safety margin for 150s limit

def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)

def _exact_maxcut(n, edges):
    """Compute exact MaxCut value via brute force for small n."""
    m = len(edges)
    if m == 0:
        return 0
    best = 0
    for mask in range(1 << n):
        cut = 0
        for (u, v) in edges:
            if (mask >> u) & 1 != (mask >> v) & 1:
                cut += 1
        if cut > best:
            best = cut
    return best

def _precompute_graph(n, edges):
    """Precompute all graph-dependent quantities for fast evaluation."""
    dim = 1 << n
    
    # Precompute Z values for each qubit: Z[i][j] = +1 if bit i of j is 0, -1 if 1
    Z = np.zeros((n, dim), dtype=np.float64)
    for i in range(n):
        for j in range(dim):
            Z[i, j] = 1.0 if ((j >> i) & 1) == 0 else -1.0
    
    # Precompute cost C(z) = sum_{(u,v) in E} (1 - Z_u Z_v) for each basis state
    m = len(edges)
    C_vals = np.zeros(dim, dtype=np.float64)
    for (u, v) in edges:
        C_vals += (1.0 - Z[u] * Z[v])
    
    # Precompute edge correlation terms for expected cut
    # For each edge, precompute Z_u * Z_v
    edge_corrs = []
    for (u, v) in edges:
        edge_corrs.append(Z[u] * Z[v])
    edge_corrs = np.array(edge_corrs)  # shape (m, dim)
    
    return Z, C_vals, edge_corrs, dim

def _apply_mixer_vec(state, beta, n, dim):
    """Apply exp(-i*beta*sum_i X_i) to state vector using iterative qubit updates."""
    c = np.cos(beta)
    s = np.sin(beta)
    state = state.copy()
    for i in range(n):
        stride = 1 << i
        period = stride * 2
        new_state = np.empty_like(state)
        for base in range(0, dim, period):
            for offset in range(stride):
                idx0 = base + offset
                idx1 = base + stride + offset
                new_state[idx0] = c * state[idx0] - 1j * s * state[idx1]
                new_state[idx1] = -1j * s * state[idx0] + c * state[idx1]
        state = new_state
    return state

def _expected_cut_vec(n, edges, gammas, betas, Z, C_vals, edge_corrs, dim):
    """
    Compute expected cut for p=2 QAOA using state-vector simulation.
    
    Circuit: |+>^n -> exp(-i*g1*C) -> exp(-i*b1*sum X) -> exp(-i*g2*C) -> exp(-i*b2*sum X)
    
    State evolution:
    |psi_0> = (1/sqrt(2^n)) * 1_vector
    |psi_1> = (1/sqrt(2^n)) * exp(-i*g1*C_vals)
    |psi_2> = mixer(b1) @ |psi_1>
    |psi_3> = |psi_2> * exp(-i*g2*C_vals)
    |psi_4> = mixer(b2) @ |psi_3>
    
    Expected cut = sum_edges (0.5 - 0.5 * <Z_u Z_v>)
    <Z_u Z_v> = sum_v |psi_4[v]|^2 * Z_u[v] * Z_v[v]
    """
    # Initial state: uniform superposition
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Apply first cost unitary
    psi *= np.exp(-1j * gammas[0] * C_vals)
    
    # Apply first mixer
    psi = _apply_mixer_vec(psi, betas[0], n, dim)
    
    # Apply second cost unitary
    psi *= np.exp(-1j * gammas[1] * C_vals)
    
    # Apply second mixer
    psi = _apply_mixer_vec(psi, betas[1], n, dim)
    
    # Compute probabilities
    probs = np.abs(psi) ** 2
    
    # Compute expected cut
    cut_val = 0.0
    for corr in edge_corrs:
        cut_val += 0.5 - 0.5 * np.sum(probs * corr)
    
    return cut_val

def _objective(params, n, edges, optimum, Z, C_vals, edge_corrs, dim):
    """Negative ratio: minimize -expected_cut/optimum."""
    if _remaining_time() < 1.0:
        return 1e10
    gammas = params[0::2]
    betas = params[1::2]
    try:
        ec = _expected_cut_vec(n, edges, gammas, betas, Z, C_vals, edge_corrs, dim)
        if optimum <= 0:
            return 0.0
        return -ec / optimum
    except:
        return 1.0

def _grid_search_angles(n, edges, optimum, Z, C_vals, edge_corrs, dim):
    """Perform a coarse-to-fine grid search over the 4D parameter space."""
    if _remaining_time() < 10.0:
        return None
    
    best_params = None
    best_val = 1e10
    
    # Stage 1: Coarse grid
    # gammas in [0, pi/2], betas in [0, pi/2]
    coarse_grid = [
        (0.0, 0.0, 0.0, 0.0),
        (0.1, 0.1, 0.1, 0.1),
        (0.2, 0.2, 0.2, 0.2),
        (0.3, 0.3, 0.3, 0.3),
        (0.4, 0.4, 0.4, 0.4),
        (0.5, 0.5, 0.5, 0.5),
        (0.6, 0.6, 0.6, 0.6),
        (0.7, 0.7, 0.7, 0.7),
        (0.8, 0.8, 0.8, 0.8),
        (0.9, 0.9, 0.9, 0.9),
        (0.1, 0.3, 0.3, 0.1),
        (0.3, 0.1, 0.1, 0.3),
        (0.1, 0.5, 0.5, 0.1),
        (0.5, 0.1, 0.1, 0.5),
        (0.2, 0.4, 0.4, 0.2),
        (0.4, 0.2, 0.2, 0.4),
    ]
    
    for params in coarse_grid:
        if _remaining_time() < 5.0:
            break
        try:
            val = _objective(np.array(params), n, edges, optimum, Z, C_vals, edge_corrs, dim)
            if val < best_val:
                best_val = val
                best_params = np.array(params)
        except:
            continue
    
    # Stage 2: Fine grid around best coarse point
    if best_params is not None and _remaining_time() > 15.0:
        center = best_params
        step = 0.1
        offsets = []
        for i in range(3):  # -step, 0, +step
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        offsets.append((i*step-step, j*step-step, k*step-step, l*step-step))
        
        for offset in offsets:
            if _remaining_time() < 5.0:
                break
            candidate = center + np.array(offset)
            # Clip to valid range
            candidate[0] = np.clip(candidate[0], 0.0, np.pi/2)
            candidate[1] = np.clip(candidate[1], 0.0, np.pi/2)
            candidate[2] = np.clip(candidate[2], 0.0, np.pi/2)
            candidate[3] = np.clip(candidate[3], 0.0, np.pi/2)
            try:
                val = _objective(candidate, n, edges, optimum, Z, C_vals, edge_corrs, dim)
                if val < best_val:
                    best_val = val
                    best_params = candidate.copy()
            except:
                continue
    
    # Stage 3: Even finer grid
    if best_params is not None and _remaining_time() > 25.0:
        center = best_params
        step = 0.03
        offsets = []
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    for l in range(5):
                        offsets.append(((i-2)*step, (j-2)*step, (k-2)*step, (l-2)*step))
        
        for offset in offsets:
            if _remaining_time() < 5.0:
                break
            candidate = center + np.array(offset)
            candidate[0] = np.clip(candidate[0], 0.0, np.pi/2)
            candidate[1] = np.clip(candidate[1], 0.0, np.pi/2)
            candidate[2] = np.clip(candidate[2], 0.0, np.pi/2)
            candidate[3] = np.clip(candidate[3], 0.0, np.pi/2)
            try:
                val = _objective(candidate, n, edges, optimum, Z, C_vals, edge_corrs, dim)
                if val < best_val:
                    best_val = val
                    best_params = candidate.copy()
            except:
                continue
    
    return best_params

def _optimize_angles(n, edges, p=2):
    """Find optimal QAOA angles for a given graph using grid search + local optimization."""
    if _remaining_time() < 5.0:
        return np.array([[0.35, 0.25], [0.25, 0.35]])
    
    optimum = _exact_maxcut(n, edges)
    if optimum <= 0:
        return np.zeros((p, 2))
    
    Z, C_vals, edge_corrs, dim = _precompute_graph(n, edges)
    
    n_params = 2 * p  # 4 parameters for p=2
    best_params = None
    best_val = 1e10
    
    # First try grid search for robust global exploration
    grid_result = _grid_search_angles(n, edges, optimum, Z, C_vals, edge_corrs, dim)
    if grid_result is not None:
        best_params = grid_result
        best_val = _objective(grid_result, n, edges, optimum, Z, C_vals, edge_corrs, dim)
    
    # Now refine with Nelder-Mead from the best grid point
    if best_params is not None and _remaining_time() > 10.0:
        try:
            result = minimize(_objective, best_params, args=(n, edges, optimum, Z, C_vals, edge_corrs, dim),
                             method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-9,
                                      'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except:
            pass
    
    # Additional polish with tighter tolerances
    if best_params is not None and _remaining_time() > 15.0:
        for _ in range(3):
            if _remaining_time() < 5.0:
                break
            try:
                result = minimize(_objective, best_params, args=(n, edges, optimum, Z, C_vals, edge_corrs, dim),
                                 method='Nelder-Mead',
                                 options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-11,
                                          'adaptive': True})
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x.copy()
            except:
                break
    
    # If grid search failed, fall back to multi-start Nelder-Mead
    if best_params is None:
        m = len(edges)
        density = m / (n * (n - 1) / 2) if n > 1 else 0
        
        inits = [
            np.array([0.3, 0.3, 0.3, 0.3]),
            np.array([0.4, 0.4, 0.4, 0.4]),
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([0.2, 0.4, 0.4, 0.2]),
            np.array([0.4, 0.2, 0.2, 0.4]),
        ]
        
        for init in inits:
            if _remaining_time() < 5.0:
                break
            try:
                result = minimize(_objective, init, args=(n, edges, optimum, Z, C_vals, edge_corrs, dim),
                                 method='Nelder-Mead',
                                 options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-8,
                                          'adaptive': True})
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x.copy()
            except:
                continue
    
    if best_params is None:
        best_params = np.array([0.35, 0.25, 0.25, 0.35])
    
    gammas = best_params[0::2]
    betas = best_params[1::2]
    return np.column_stack([gammas, betas])

def angles(n, edges, p):
    """Main entry point: return QAOA angles for the given graph."""
    if _remaining_time() < 2.0:
        return np.array([[0.35, 0.25], [0.25, 0.35]])
    
    try:
        result = _optimize_angles(n, edges, p)
        if result.shape != (p, 2):
            result = np.tile([0.35, 0.25], (p, 1))
        return result
    except:
        return np.array([[0.35, 0.25], [0.25, 0.35]])
# EVOLVE-BLOCK-END
