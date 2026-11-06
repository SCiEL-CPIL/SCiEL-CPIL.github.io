"""QAOA parameter design for MaxCut on small graphs."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

_DEADLINE = 150.0
_start_time = None
_cache = {}


def _get_deadline():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time + _DEADLINE


def _time_left():
    return _get_deadline() - time.time()


def _build_cost_matrix(n, edges):
    """Build the cost matrix C where C[i,j] = 1 if (i,j) is an edge."""
    C = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        C[u, v] = 1.0
        C[v, u] = 1.0
    return C


def _exact_optimal_cut(n, edges):
    """Compute exact optimal MaxCut value by brute force."""
    best = 0
    for bits in range(2 ** n):
        cut = 0
        for u, v in edges:
            bu = (bits >> u) & 1
            bv = (bits >> v) & 1
            if bu != bv:
                cut += 1
        if cut > best:
            best = cut
    return best


def _qaoa_expected_cut(n, edges, gammas, betas):
    """
    Compute expected cut value for QAOA with given parameters.
    Uses exact state vector simulation.
    """
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have same length")
    p = len(gammas)
    
    # Initial state: |+>^n = (1/sqrt(2^n)) * ones(2^n)
    dim = 2 ** n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute edge list for cost
    edge_list = list(edges)
    
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply cost unitary: exp(-i * gamma * C)
        # C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
        # exp(-i gamma C) = prod_{(u,v) in E} exp(-i gamma (1 - Z_u Z_v)/2)
        # = prod_{(u,v) in E} [exp(-i gamma/2) * exp(i gamma/2 * Z_u Z_v)]
        # = exp(-i gamma * |E|/2) * prod_{(u,v) in E} exp(i gamma/2 * Z_u Z_v)
        # Since Z_u Z_v is diagonal with eigenvalues +/-1:
        # exp(i gamma/2 * Z_u Z_v) is diagonal with entries exp(i gamma/2) if Z_u Z_v = +1, 
        # exp(-i gamma/2) if Z_u Z_v = -1
        
        phase = np.exp(-1j * gamma * len(edge_list) / 2.0)
        diag = np.ones(dim, dtype=np.complex128)
        
        for u, v in edge_list:
            # Z_u Z_v: for each basis state, check if bits u and v are same
            # If same -> +1, else -> -1
            # Z_u Z_v eigenvalue = 1 if bit_u == bit_v, -1 otherwise
            # exp(i gamma/2 * Z_u Z_v) = exp(i gamma/2) if same, exp(-i gamma/2) if different
            for idx in range(dim):
                bu = (idx >> u) & 1
                bv = (idx >> v) & 1
                if bu == bv:
                    diag[idx] *= np.exp(1j * gamma / 2.0)
                else:
                    diag[idx] *= np.exp(-1j * gamma / 2.0)
        
        state = phase * diag * state
        
        # Apply mixing unitary: exp(-i beta * sum X_j)
        # = prod_j exp(-i beta X_j)
        # exp(-i beta X) = cos(beta) I - i sin(beta) X
        # X flips the bit
        for j in range(n):
            cos_b = np.cos(beta)
            sin_b = np.sin(beta)
            new_state = np.zeros(dim, dtype=np.complex128)
            for idx in range(dim):
                flipped = idx ^ (1 << j)
                new_state[idx] = cos_b * state[idx] - 1j * sin_b * state[flipped]
            state = new_state
    
    # Compute expected cut: <state| C |state>
    # C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
    # <C> = sum_{(u,v) in E} (1 - <Z_u Z_v>)/2
    # <Z_u Z_v> = sum_idx |state[idx]|^2 * z_u(idx) * z_v(idx)
    # where z_j(idx) = 1 if bit_j=0, -1 if bit_j=1
    
    prob = np.abs(state) ** 2
    expected_cut = 0.0
    for u, v in edge_list:
        zz = 0.0
        for idx in range(dim):
            zu = 1.0 if ((idx >> u) & 1) == 0 else -1.0
            zv = 1.0 if ((idx >> v) & 1) == 0 else -1.0
            zz += prob[idx] * zu * zv
        expected_cut += (1.0 - zz) / 2.0
    
    return expected_cut


def _qaoa_expected_cut_fast(n, edges, gammas, betas):
    """
    Faster computation using matrix operations where possible.
    For small n, the loop version is fine, but let's optimize the inner loops.
    """
    p = len(gammas)
    dim = 2 ** n
    edge_list = list(edges)
    m = len(edge_list)
    
    # Precompute Z_u Z_v for each edge as a vector of +1/-1
    zz_vectors = []
    for u, v in edge_list:
        zz = np.ones(dim, dtype=np.float64)
        for idx in range(dim):
            bu = (idx >> u) & 1
            bv = (idx >> v) & 1
            if bu != bv:
                zz[idx] = -1.0
        zz_vectors.append(zz)
    
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Cost unitary (diagonal)
        phase = np.exp(-1j * gamma * m / 2.0)
        diag = np.ones(dim, dtype=np.complex128)
        for zz in zz_vectors:
            # exp(i gamma/2 * Z_u Z_v)
            # where zz[idx] is +1 or -1
            diag *= np.exp(1j * gamma / 2.0 * zz)
        state = phase * diag * state
        
        # Mixing unitary
        for j in range(n):
            cos_b = np.cos(beta)
            sin_b = np.sin(beta)
            flipped = np.array([(idx ^ (1 << j)) for idx in range(dim)], dtype=np.int64)
            state = cos_b * state - 1j * sin_b * state[flipped]
    
    # Expected cut
    prob = np.abs(state) ** 2
    expected_cut = 0.0
    for zz in zz_vectors:
        zz_exp = np.sum(prob * zz)
        expected_cut += (1.0 - zz_exp) / 2.0
    
    return expected_cut


def _optimize_params(n, edges, p=2, max_iter=200, tol=1e-10):
    """
    Optimize QAOA parameters using Nelder-Mead simplex method.
    Returns (gammas, betas) and the best expected cut value.
    """
    from scipy.optimize import minimize
    
    best_cut = _exact_optimal_cut(n, edges)
    
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        # Clip to reasonable range
        gammas = np.clip(gammas, -np.pi, np.pi)
        betas = np.clip(betas, -np.pi, np.pi)
        val = _qaoa_expected_cut_fast(n, edges, gammas, betas)
        return -val  # Minimize negative to maximize
    
    # Multi-start with different initial points
    best_result = None
    best_val = -np.inf
    
    np.random.seed(42)
    n_starts = 8
    for s in range(n_starts):
        if _time_left() < 5.0:
            break
        if s == 0:
            x0 = np.array([0.5, 0.5, 0.5, 0.5])  # Common initial guess
        else:
            x0 = np.random.uniform(0, np.pi, size=2 * p)
        
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
        
        if res.fun < best_val:
            best_val = res.fun
            best_result = res.x
    
    if best_result is None:
        # Fallback
        gammas = np.array([0.5, 0.5])
        betas = np.array([0.5, 0.5])
    else:
        gammas = np.clip(best_result[:p], -np.pi, np.pi)
        betas = np.clip(best_result[p:], -np.pi, np.pi)
    
    best_expected = _qaoa_expected_cut_fast(n, edges, gammas, betas)
    return gammas, betas, best_expected


def _graph_key(n, edges):
    """Create a hashable key for the graph."""
    return (n, tuple(sorted(tuple(sorted(e)) for e in edges)))


def angles(n, edges, p):
    """
    Return QAOA angles (p, 2) array with rows (gamma, beta) in radians.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    if p != 2:
        # Handle general p by repeating or truncating
        # For now, focus on p=2
        pass
    
    key = _graph_key(n, edges)
    
    if key in _cache:
        return _cache[key]
    
    # Check time
    if _time_left() < 10.0:
        # Not enough time, return default
        result = np.tile([0.35, 0.25], (p, 1))
        _cache[key] = result
        return result
    
    try:
        gammas, betas, best_expected = _optimize_params(n, edges, p=p)
        result = np.column_stack([gammas, betas])
        _cache[key] = result
        return result
    except Exception:
        # Fallback on error
        result = np.tile([0.35, 0.25], (p, 1))
        _cache[key] = result
        return result
# EVOLVE-BLOCK-END
