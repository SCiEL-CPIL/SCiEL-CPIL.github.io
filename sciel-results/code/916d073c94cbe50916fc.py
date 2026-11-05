"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Cache for exact maxcut values
_exact_maxcut_cache = {}
# Cache for best angles found per graph signature
_angles_cache = {}
# Cache for precomputed graph data
_graph_data_cache = {}

def _graph_signature(n, edges):
    """Create a hashable signature for the graph."""
    edge_set = frozenset(frozenset((int(u), int(v))) for u, v in edges)
    return (n, edge_set)

def _compute_edge_bitmasks(n, edges):
    """
    Precompute bitmasks for each edge. For edge (u,v), the mask has bits u and v set.
    """
    masks = []
    for u, v in edges:
        mask = (1 << int(u)) | (1 << int(v))
        masks.append(mask)
    return masks

def _cost_vector_fast(n, edges):
    """
    Compute the cost vector c[x] = sum of edges cut by bitstring x using vectorized ops.
    """
    m = 1 << n
    edge_masks = _compute_edge_bitmasks(n, edges)
    c = np.zeros(m, dtype=np.float64)
    for mask in edge_masks:
        # Edge is cut if exactly one of the two bits is set
        # (x & mask) has exactly one bit set iff (x & mask) != 0 and (x & mask) & ((x & mask) - 1) == 0
        # More efficient: cut = 1 if (x_u XOR x_v) == 1
        # x_u = (x >> u) & 1, x_v = (x >> v) & 1
        # We can extract u and v from mask
        u = int(np.log2(mask & (-mask)))  # lowest set bit
        v = int(np.log2(mask >> u))       # next set bit after removing lowest
        # Actually simpler: use bit operations
        # For each x, edge is cut iff (x >> u) & 1 != (x >> v) & 1
        idx = np.arange(m)
        xu = (idx >> u) & 1
        xv = (idx >> v) & 1
        c += (xu != xv).astype(np.float64)
    return c

def _exact_maxcut(n, edges):
    """Compute the exact maximum cut value for the graph."""
    sig = _graph_signature(n, edges)
    if sig in _exact_maxcut_cache:
        return _exact_maxcut_cache[sig]
    
    c = _cost_vector_fast(n, edges)
    opt = float(np.max(c))
    _exact_maxcut_cache[sig] = opt
    return opt

def _precompute_mixer_indices(n):
    """
    Precompute index pairs for each qubit's mixer operation.
    Returns a list of (idx_zero, idx_one) arrays for each qubit.
    """
    m = 1 << n
    idx = np.arange(m)
    pairs = []
    for q in range(n):
        bit = 1 << q
        mask_zero = (idx & bit) == 0
        idx_zero = idx[mask_zero]
        idx_one = idx_zero | bit
        pairs.append((idx_zero, idx_one))
    return pairs

def _qaoa_forward(n, c, mixer_pairs, gammas, betas):
    """
    Fast QAOA forward pass. Returns the final state vector.
    """
    m = 1 << n
    psi = np.ones(m, dtype=np.complex128) / np.sqrt(m)
    
    for k in range(len(gammas)):
        # Apply cost phase
        psi *= np.exp(-1j * gammas[k] * c)
        
        # Apply mixer
        beta = betas[k]
        cb = np.cos(beta)
        sb = np.sin(beta)
        
        for q in range(n):
            idx_zero, idx_one = mixer_pairs[q]
            psi_zero = psi[idx_zero].copy()
            psi_one = psi[idx_one].copy()
            psi[idx_zero] = cb * psi_zero - 1j * sb * psi_one
            psi[idx_one] = -1j * sb * psi_zero + cb * psi_one
    
    return psi

def _qaoa_expectation(n, c, mixer_pairs, gammas, betas):
    """Compute expected cut value."""
    psi = _qaoa_forward(n, c, mixer_pairs, gammas, betas)
    probs = np.abs(psi) ** 2
    return np.dot(probs, c)

def _qaoa_grad(n, c, mixer_pairs, gammas, betas):
    """
    Compute gradient of expected cut using parameter-shift rule.
    For cost operator (diagonal), gradient wrt gamma_k uses shift pi/2.
    For mixer (X operators), gradient wrt beta_k uses shift pi/2.
    
    Returns (dE/dgamma_k, dE/dbeta_k) for each k.
    """
    p = len(gammas)
    dE = np.zeros(2 * p)
    
    for k in range(p):
        # Gradient wrt gamma_k: shift by pi/2
        g_plus = gammas.copy()
        g_minus = gammas.copy()
        g_plus[k] += np.pi / 2
        g_minus[k] -= np.pi / 2
        
        E_plus = _qaoa_expectation(n, c, mixer_pairs, g_plus, betas)
        E_minus = _qaoa_expectation(n, c, mixer_pairs, g_minus, betas)
        dE[2 * k] = (E_plus - E_minus) / 2.0  # dE/dgamma_k
        
        # Gradient wrt beta_k: shift by pi/2
        b_plus = betas.copy()
        b_minus = betas.copy()
        b_plus[k] += np.pi / 2
        b_minus[k] -= np.pi / 2
        
        E_plus = _qaoa_expectation(n, c, mixer_pairs, gammas, b_plus)
        E_minus = _qaoa_expectation(n, c, mixer_pairs, gammas, b_minus)
        dE[2 * k + 1] = (E_plus - E_minus) / 2.0  # dE/dbeta_k
    
    return dE

def _objective_grad(params, n, c, mixer_pairs):
    """
    Objective (negative expected cut) and gradient for L-BFGS-B.
    params: flat array [gamma_0, beta_0, gamma_1, beta_1, ...]
    """
    p = len(params) // 2
    gammas = params[0::2]
    betas = params[1::2]
    
    E = _qaoa_expectation(n, c, mixer_pairs, gammas, betas)
    grad_E = _qaoa_grad(n, c, mixer_pairs, gammas, betas)
    
    # Gradient of -E is -grad_E
    grad_neg = -grad_E
    return -E, grad_neg

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio
    of the independently simulated expected cut to the exact classical optimum.
    
    Returns: real array (p, 2), rows (gamma, beta) in radians.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    sig = _graph_signature(n, edges)
    
    # Check cache
    if sig in _angles_cache:
        return _angles_cache[sig]
    
    # Compute exact maxcut
    opt_cut = _exact_maxcut(n, edges)
    
    if opt_cut == 0:
        result = np.zeros((p, 2), dtype=np.float64)
        _angles_cache[sig] = result
        return result
    
    # Precompute graph data
    if sig in _graph_data_cache:
        c, mixer_pairs = _graph_data_cache[sig]
    else:
        c = _cost_vector_fast(n, edges)
        mixer_pairs = _precompute_mixer_indices(n)
        _graph_data_cache[sig] = (c, mixer_pairs)
    
    # Multi-start L-BFGS-B optimization with analytical gradient
    best_params = None
    best_value = -np.inf
    
    # Initial guesses - diverse starts in [0, pi] range
    initial_guesses = [
        np.array([0.35, 0.25] * p, dtype=np.float64),
        np.array([0.5, 0.3] * p, dtype=np.float64),
        np.array([0.25, 0.15] * p, dtype=np.float64),
        np.array([0.7, 0.4] * p, dtype=np.float64),
        np.array([0.1, 0.1] * p, dtype=np.float64),
        np.array([0.9, 0.5] * p, dtype=np.float64),
        np.array([0.4, 0.35] * p, dtype=np.float64),
        np.array([0.6, 0.2] * p, dtype=np.float64),
    ]
    
    # Add random starts with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    for _ in range(6):
        initial_guesses.append(rng.uniform(0.05, 1.5, size=2 * p))
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _objective_grad,
                guess,
                args=(n, c, mixer_pairs),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, np.pi)] * (2 * p),
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            val = result.fun
            if val < best_value:
                best_value = val
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    if best_params is None:
        result = np.tile([0.35, 0.25], (p, 1))
    else:
        gammas = best_params[0::2]
        betas = best_params[1::2]
        result = np.column_stack((gammas, betas))
    
    _angles_cache[sig] = result
    return result
# EVOLVE-BLOCK-END
