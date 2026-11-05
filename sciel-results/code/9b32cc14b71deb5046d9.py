"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache

_cache = {}

def _vectorized_evaluate(n, edges, gammas, betas):
    """
    Fast vectorized QAOA expected cut evaluation.
    Uses precomputed bit patterns to avoid Python loops over basis states.
    """
    dim = 1 << n
    m = len(edges)
    
    if m == 0:
        return 0.0
    
    # Precompute bit patterns for all basis states (vectorized)
    # bits[i, k] = (i >> k) & 1
    bits = np.array([[(idx >> k) & 1 for k in range(n)] for idx in range(dim)], dtype=np.int8)
    
    # Precompute for each edge: (-1)^(bit_u + bit_v) for all basis states
    # This is a (dim, m) matrix
    edge_parity = np.zeros((dim, m), dtype=np.float64)
    for j, (u, v) in enumerate(edges):
        # (-1)^(bit_u + bit_v) = 1 if bits equal, -1 if different
        edge_parity[:, j] = np.where(bits[:, u] == bits[:, v], 1.0, -1.0)
    
    # Also precompute C eigenvalues: sum over edges of (-1)^(bit_u + bit_v)
    c_eigenvalues = edge_parity.sum(axis=1)  # shape (dim,)
    
    # Start state: |+>^n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute cos/sin for each layer
    for l in range(len(gammas)):
        gamma = gammas[l]
        beta = betas[l]
        
        # Apply exp(-i * gamma * C): diagonal phase
        psi *= np.exp(-1j * gamma * c_eigenvalues)
        
        # Apply exp(-i * beta * sum X_i)
        # This is a tensor product of single-qubit rotations
        # exp(-i*beta*X) = cos(beta)*I - i*sin(beta)*X
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Apply to each qubit using reshape trick
        psi_tensor = psi.reshape([2] * n)
        
        for q in range(n):
            # For qubit q: new[0,...] = cos*old[0,...] - i*sin*old[1,...]
            #             new[1,...] = cos*old[1,...] - i*sin*old[0,...]
            slc0 = [slice(None)] * n
            slc0[q] = 0
            slc1 = [slice(None)] * n
            slc1[q] = 1
            slc0 = tuple(slc0)
            slc1 = tuple(slc1)
            
            tmp0 = psi_tensor[slc0].copy()
            tmp1 = psi_tensor[slc1].copy()
            
            psi_tensor[slc0] = cos_b * tmp0 - 1j * sin_b * tmp1
            psi_tensor[slc1] = cos_b * tmp1 - 1j * sin_b * tmp0
        
        psi = psi_tensor.reshape(-1)
    
    # Compute expected cut
    probs = np.abs(psi) ** 2  # shape (dim,)
    
    # For each edge j: <Z_u Z_v> = sum_i probs[i] * edge_parity[i, j]
    zz_expects = edge_parity.T @ probs  # shape (m,)
    
    expected_cut = np.sum((1.0 - zz_expects) / 2.0)
    return float(expected_cut)


def _exhaustive_maxcut_fast(n, edges):
    """Fast exhaustive MaxCut using vectorized operations."""
    if not edges:
        return 0
    
    dim = 1 << n
    # For each assignment mask, compute cut size
    # Vectorized: for each edge (u,v), check if bits differ
    edges_arr = np.array(edges)  # shape (m, 2)
    
    # Create all masks
    masks = np.arange(dim)
    
    # Extract bits for all edges
    # bits_for_edge_u = (masks >> edges[:, 0]) & 1
    # This needs broadcasting: masks[:, None] >> edges[None, :, 0] & 1
    u_bits = (masks[:, None] >> edges_arr[None, :, 0]) & 1  # shape (dim, m)
    v_bits = (masks[:, None] >> edges_arr[None, :, 1]) & 1  # shape (dim, m)
    
    cut_sizes = np.sum(u_bits != v_bits, axis=1)
    return int(cut_sizes.max())


def _optimize_fast(n, edges, p=2):
    """
    Multi-start L-BFGS-B optimization with fast vectorized evaluation.
    """
    from scipy.optimize import minimize
    
    best_maxcut = _exhaustive_maxcut_fast(n, edges)
    if best_maxcut == 0:
        return np.zeros((p, 2))
    
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        exp_cut = _vectorized_evaluate(n, edges, gammas, betas)
        return -exp_cut / best_maxcut
    
    bounds = [(0.0, np.pi)] * (2 * p)
    
    # Multiple starting points
    best_result = None
    best_val = np.inf
    
    # Generate diverse starting points
    rng = np.random.RandomState(42)
    start_points = []
    
    # Known good initializations for p=2
    # Based on literature: gamma ~ 0.5*sqrt(pi*ln(2))/p, beta ~ 0.5*pi/(2*p)
    base_g = 0.5 * np.sqrt(np.pi * np.log(2)) / p
    base_b = 0.5 * np.pi / (2 * p)
    
    start_points.append(np.array([base_g] * p + [base_b] * p))
    start_points.append(np.array([0.3] * p + [0.2] * p))
    start_points.append(np.array([0.5] * p + [0.3] * p))
    start_points.append(np.array([0.2] * p + [0.1] * p))
    
    # Add random starts
    for _ in range(5):
        gs = rng.uniform(0.1, 1.0, size=p)
        bs = rng.uniform(0.05, 0.5, size=p)
        start_points.append(np.concatenate([gs, bs]))
    
    for x0 in start_points:
        x0 = np.clip(x0, 0.01, np.pi - 0.01)
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                             bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            val = result.fun
            if val < best_val:
                best_val = val
                best_result = result.x
        except Exception:
            continue
    
    if best_result is None:
        return np.tile([0.35, 0.25], (p, 1))
    
    params = np.clip(best_result, 0, np.pi)
    gammas = params[:p]
    betas = params[p:]
    return np.column_stack([gammas, betas])


def angles(n, edges, p):
    """
    Compute QAOA angles for the given graph using fast vectorized optimization.
    Results are cached for repeated calls with the same graph.
    """
    edges_list = tuple(sorted((int(u), int(v)) for u, v in edges))
    cache_key = (n, edges_list, p)
    
    if cache_key in _cache:
        return _cache[cache_key].copy()
    
    try:
        opt_angles = _optimize_fast(n, list(edges_list), p=p)
        opt_angles = opt_angles.reshape(p, 2)
        opt_angles = np.clip(opt_angles, 0, np.pi)
        _cache[cache_key] = opt_angles
        return opt_angles
    except Exception:
        fallback = np.tile([0.35, 0.25], (p, 1))
        return fallback
# EVOLVE-BLOCK-END
