"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))


def _delta_swap(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in the assignment p.
    """
    pi = p[i]
    pj = p[j]
    
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    
    diff1 = b[pj, p[mask]] - b[pi, p[mask]]
    diff2 = b[p[mask], pj] - b[p[mask], pi]
    
    a_diff_row = a[i, mask] - a[j, mask]
    a_diff_col = a[mask, i] - a[mask, j]
    
    delta_k = np.sum(a_diff_row * diff1) + np.sum(a_diff_col * diff2)
    
    delta_self = (a[i, j] * (b[pj, pi] - b[pi, pj])
                  + a[j, i] * (b[pi, pj] - b[pj, pi])
                  + a[i, i] * (b[pj, pj] - b[pi, pi])
                  + a[j, j] * (b[pi, pi] - b[pj, pj]))
    
    return int(delta_k + delta_self)


def _local_search(a, b, p, deadline):
    """Best-improvement 2-opt local search with time guard."""
    n = len(p)
    cost = _cost(a, b, p)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
    return p.copy(), cost


def _spectral_init(a, b):
    """Spectral initialization: embed facilities and locations, match by nearest neighbor."""
    n = a.shape[0]
    # Build Laplacian-like matrices for spectral embedding
    # For facilities: use flow matrix a
    # For locations: use distance matrix b
    
    # Symmetrize for eigenvalue computation
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # Compute eigenvectors
    try:
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
        
        # Use smallest eigenvectors (most "central" embedding)
        # Take top k eigenvectors for embedding
        k = min(10, n)
        # For facilities, use eigenvectors corresponding to smallest eigenvalues
        idx_a = np.argsort(evals_a)
        fac_emb = evecs_a[:, idx_a[:k]]
        
        # For locations, use eigenvectors corresponding to largest eigenvalues (most spread out)
        idx_b = np.argsort(-evals_b)
        loc_emb = evecs_b[:, idx_b[:k]]
        
        # Normalize embeddings
        fac_norm = np.linalg.norm(fac_emb, axis=1, keepdims=True)
        fac_norm[fac_norm == 0] = 1
        fac_emb = fac_emb / fac_norm
        
        loc_norm = np.linalg.norm(loc_emb, axis=1, keepdims=True)
        loc_norm[loc_norm == 0] = 1
        loc_emb = loc_emb / loc_norm
        
        # Match by negative inner product (closest embeddings)
        cost_matrix = -fac_emb @ loc_emb.T
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    except:
        # Fallback to rank-matching
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin

    best_p = None
    best_cost = np.inf

    def _update_best(p, cost):
        nonlocal best_p, best_cost
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()

    # 1. Spectral Initialization
    if time.time() < deadline - 10:
        p_spec = _spectral_init(a, b)
        cost_spec = _cost(a, b, p_spec)
        _update_best(p_spec, cost_spec)
        
        if time.time() < deadline - 5:
            p_ls, cost_ls = _local_search(a, b, p_spec.copy(), deadline - 5)
            _update_best(p_ls, cost_ls)

    # 2. Linear Assignment Initialization (Surrogate)
    if time.time() < deadline - 10:
        surrogate = a @ b + a.T @ b
        row_ind, col_ind = linear_sum_assignment(surrogate)
        p_lap = np.empty(n, dtype=np.int64)
        p_lap[row_ind] = col_ind
        cost_lap = _cost(a, b, p_lap)
        _update_best(p_lap, cost_lap)

        if time.time() < deadline - 5:
            p_ls, cost_ls = _local_search(a, b, p_lap.copy(), deadline - 5)
            _update_best(p_ls, cost_ls)

    # 3. Iterated Local Search (ILS) from best solution
    if best_p is not None and time.time() < deadline - 5:
        current_p = best_p.copy()
        current_cost = best_cost
        rng = np.random.default_rng(42)
        
        while time.time() < deadline - 2:
            # Perturb current solution
            # Number of swaps for perturbation
            n_swaps = rng.integers(2, 5)
            p_perturbed = current_p.copy()
            for _ in range(n_swaps):
                i, j = rng.integers(0, n, 2)
                if i != j:
                    p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
            
            # Local search on perturbed solution
            p_ls, cost_ls = _local_search(a, b, p_perturbed, deadline - 2)
            
            # Accept if better
            if cost_ls < current_cost:
                current_p = p_ls
                current_cost = cost_ls
                _update_best(current_p, current_cost)
            else:
                # Keep current_p as the base for next perturbation 
                pass

    if best_p is None:
        best_p = np.arange(n)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
