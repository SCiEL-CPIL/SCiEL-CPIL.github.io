"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache

_cache = {}

def _vectorized_evaluate(n, edges, gammas, betas):
    """
    Fast vectorized QAOA expected cut evaluation.
    """
    dim = 1 << n
    m = len(edges)
    
    if m == 0:
        return 0.0
    
    # Precompute bit patterns for all basis states
    # bits[i, k] = (i >> k) & 1
    # For n=10, dim=1024, this is small.
    # We can compute edge_parity on the fly or cache it per graph.
    # Since edges are fixed per call, and we cache results per graph, 
    # we should precompute edge_parity and c_eigenvalues outside the loop 
    # if we restructure. But for simplicity and to keep the function 
    # self-contained for the optimizer, we compute here.
    # Optimization: Precompute edge_parity once per graph in a helper?
    # No, the optimizer calls this many times. We should pass precomputed data.
    
    # Let's create a helper that returns the evaluator function with precomputed data.
    pass 

# To make it faster, we will precompute the "static" parts of the graph 
# (edge parities, etc.) and pass them to the evaluation function.

def _prepare_graph_data(n, edges):
    """
    Precompute data needed for fast evaluation.
    Returns:
        edge_parity: (dim, m) array where entry is 1 if bits same, -1 if diff.
        c_eigenvalues: (dim,) array of C eigenvalues.
        dim: 2^n
    """
    dim = 1 << n
    m = len(edges)
    if m == 0:
        return None, None, dim
        
    # bits[i, k] = (i >> k) & 1
    # Instead of building a full bits matrix (dim, n), we can compute edge parity directly.
    # edge_parity[i, j] = 1 if bit_u == bit_v else -1
    # bit_u = (i >> u) & 1
    # bit_v = (i >> v) & 1
    
    indices = np.arange(dim)
    edge_parity = np.zeros((dim, m), dtype=np.float64)
    
    for j, (u, v) in enumerate(edges):
        bu = (indices >> u) & 1
        bv = (indices >> v) & 1
        # If bu == bv, parity is 1 (same side, no cut). If bu != bv, parity is -1 (cut).
        # Wait, C = sum Z_u Z_v. 
        # Z_u Z_v eigenvalue is (+1) if same, (-1) if different.
        # So edge_parity is the eigenvalue of Z_u Z_v.
        edge_parity[:, j] = np.where(bu == bv, 1.0, -1.0)
        
    c_eigenvalues = edge_parity.sum(axis=1)
    return edge_parity, c_eigenvalues, dim

def _make_evaluator(n, edge_parity, c_eigenvalues, dim):
    """
    Create a fast evaluation closure.
    """
    def evaluate(gammas, betas):
        # psi starts as |+>^n
        # psi = [1/sqrt(dim)] * dim
        # We can avoid storing psi as complex if we just track probabilities?
        # No, phases matter for interference.
        
        psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        
        for l in range(len(gammas)):
            gamma = gammas[l]
            beta = betas[l]
            
            # Apply exp(-i gamma C)
            # psi *= exp(-1j * gamma * c_eigenvalues)
            psi *= np.exp(-1j * gamma * c_eigenvalues)
            
            # Apply exp(-i beta sum X)
            cos_b = np.cos(beta)
            sin_b = np.sin(beta)
            
            # Apply X rotation on each qubit
            # This is the slow part. 
            # For n=10, dim=1024.
            # Reshape to [2, 2, ..., 2]
            psi_tensor = psi.reshape([2] * n)
            
            for q in range(n):
                slc0 = [slice(None)] * n
                slc0[q] = 0
                slc0 = tuple(slc0)
                slc1 = [slice(None)] * n
                slc1[q] = 1
                slc1 = tuple(slc1)
                
                tmp0 = psi_tensor[slc0].copy()
                tmp1 = psi_tensor[slc1].copy()
                
                psi_tensor[slc0] = cos_b * tmp0 - 1j * sin_b * tmp1
                psi_tensor[slc1] = cos_b * tmp1 - 1j * sin_b * tmp0
                
            psi = psi_tensor.reshape(-1)
            
        probs = np.abs(psi) ** 2
        zz_expects = edge_parity.T @ probs
        expected_cut = np.sum((1.0 - zz_expects) / 2.0)
        return float(expected_cut)
        
    return evaluate

def _exhaustive_maxcut_fast(n, edges):
    """Fast exhaustive MaxCut using vectorized operations."""
    if not edges:
        return 0
    
    dim = 1 << n
    edges_arr = np.array(edges)
    
    masks = np.arange(dim)
    
    u_bits = (masks[:, None] >> edges_arr[None, :, 0]) & 1
    v_bits = (masks[:, None] >> edges_arr[None, :, 1]) & 1
    
    cut_sizes = np.sum(u_bits != v_bits, axis=1)
    return int(cut_sizes.max())

def _optimize_de(n, edges, p=2):
    """
    Differential Evolution optimization for QAOA angles.
    """
    from scipy.optimize import differential_evolution
    
    best_maxcut = _exhaustive_maxcut_fast(n, edges)
    if best_maxcut == 0:
        return np.zeros((p, 2))
        
    edge_parity, c_eigenvalues, dim = _prepare_graph_data(n, edges)
    if edge_parity is None:
        return np.zeros((p, 2))
        
    evaluator = _make_evaluator(n, edge_parity, c_eigenvalues, dim)
    
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        exp_cut = evaluator(gammas, betas)
        return -exp_cut  # Maximize exp_cut
        
    bounds = [(0.0, np.pi)] * (2 * p)
    
    # Differential Evolution parameters
    # popsize: 15 * dim = 15 * 4 = 60 individuals
    # tol: 1e-5
    # maxiter: 100
    # seed: 42 for reproducibility
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            tol=1e-6, 
            popsize=20,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False  # We will do our own polish or just rely on DE
        )
        
        if result.success or result.fun < 0:
            best_params = result.x
        else:
            # Fallback to a good heuristic if DE fails
            best_params = np.array([0.5, 0.5, 0.3, 0.3])[:2*p]
            
    except Exception:
        best_params = np.array([0.5, 0.5, 0.3, 0.3])[:2*p]
        
    # Optional: Polish with a few Nelder-Mead steps?
    # DE is usually quite accurate. Let's add a simple local polish using Nelder-Mead
    # to squeeze out a bit more precision.
    
    from scipy.optimize import minimize
    
    try:
        polish_result = minimize(
            objective, 
            best_params, 
            method='Nelder-Mead', 
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        if polish_result.fun < result.fun if hasattr(result, 'fun') else True:
             # Check if polish improved it
             # result.fun might not exist if exception, so handle carefully
             if not hasattr(result, 'fun') or polish_result.fun < result.fun:
                 best_params = polish_result.x
    except Exception:
        pass
        
    params = np.clip(best_params, 0, np.pi)
    gammas = params[:p]
    betas = params[p:]
    return np.column_stack([gammas, betas])

def angles(n, edges, p):
    """
    Compute QAOA angles for the given graph using Differential Evolution.
    """
    edges_list = tuple(sorted((int(u), int(v)) for u, v in edges))
    cache_key = (n, edges_list, p)
    
    if cache_key in _cache:
        return _cache[cache_key].copy()
    
    try:
        opt_angles = _optimize_de(n, list(edges_list), p=p)
        opt_angles = opt_angles.reshape(p, 2)
        opt_angles = np.clip(opt_angles, 0, np.pi)
        _cache[cache_key] = opt_angles
        return opt_angles
    except Exception:
        fallback = np.tile([0.35, 0.25], (p, 1))
        return fallback
# EVOLVE-BLOCK-END
