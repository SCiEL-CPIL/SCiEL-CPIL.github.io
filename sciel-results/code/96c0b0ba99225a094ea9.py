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


def _local_search(a, b, p, deadline, surrogate=None):
    """First-improvement 2-opt local search with time guard and optional surrogate-guided move ordering."""
    n = len(p)
    cost = _cost(a, b, p)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # If we have a surrogate, prioritize moves that look good in surrogate space
            if surrogate is not None:
                # Compute surrogate delta for each j > i, sort by surrogate improvement
                j_candidates = list(range(i + 1, n))
                if len(j_candidates) > 50:
                    # For large n, sample to keep it fast
                    sample_idx = np.random.choice(len(j_candidates), size=50, replace=False)
                    j_candidates = [j_candidates[k] for k in sample_idx]
                surr_deltas = []
                for j in j_candidates:
                    pi, pj = p[i], p[j]
                    surr_d = surrogate[i, pj] - surrogate[i, pi] + surrogate[j, pi] - surrogate[j, pj]
                    surr_deltas.append((surr_d, j))
                surr_deltas.sort(key=lambda x: x[0])
                j_candidates = [j for _, j in surr_deltas]
            else:
                j_candidates = list(range(i + 1, n))
            
            for j in j_candidates:
                if time.time() > deadline:
                    break
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    break  # First improvement per i, then restart scan
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


def _build_surrogate(a, b, p):
    """
    Build a Benders-style surrogate cost matrix C[i, l] approximating
    the marginal cost of assigning facility i to location l.
    
    C[i, l] = sum_j a[i, j] * b[l, p[j]] + sum_j a[j, i] * b[p[j], l]
    This is the exact linearization of the QAP cost when only facility i's
    assignment is varied, holding others fixed.
    """
    n = len(p)
    # For each facility i and location l:
    # C[i, l] = sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l]
    #          = (a @ B_p)[i, l] + (a.T @ B_p.T)[i, l]  where B_p = b[np.ix_(p,p)]
    # Actually: sum_j a[i,j] * b[l, p[j]] = (a @ b[np.ix_(p, p)].T)[i, l]
    #           sum_j a[j,i] * b[p[j], l] = (a.T @ b[np.ix_(p, p)])[i, l]
    Bp = b[np.ix_(p, p)]
    # First term: for each i, sum over j of a[i,j]*b[l,p[j]] for each l
    # = a @ Bp.T  (n x n) where (a @ Bp.T)[i,l] = sum_j a[i,j]*Bp.T[j,l] = sum_j a[i,j]*Bp[l,j] = sum_j a[i,j]*b[p[j],p[l]]
    # Wait, that's not quite right. Let me recompute.
    # We want C[i,l] = sum_j a[i,j] * b[l, p[j]]
    # Let Bp[j,k] = b[p[j], p[k]]. Then b[l, p[j]] is not directly Bp.
    # b[l, p[j]]: l is a location index, p[j] is a location index.
    # So sum_j a[i,j] * b[l, p[j]] = (a @ b[:, p])[i, l] where b[:, p] is b with columns indexed by p
    Bp_col = b[:, p]  # shape (n, n), Bp_col[j, k] = b[p[k], j]... no
    # b[:, p] has shape (n, n) where (b[:, p])[row, col] = b[row, p[col]]
    # So (b[:, p])[l, j] = b[l, p[j]]
    # sum_j a[i,j] * b[l, p[j]] = sum_j a[i,j] * (b[:, p])[l, j] = (a @ (b[:, p]).T)[i, l]
    term1 = a @ b[:, p].T  # (n,n), term1[i,l] = sum_j a[i,j]*b[l,p[j]]
    # Second term: sum_j a[j,i] * b[p[j], l]
    # b[p[j], l] = (b[p, :])[j, l]
    # sum_j a[j,i] * (b[p, :])[j, l] = (a.T @ b[p, :])[i, l]
    term2 = a.T @ b[p, :]  # (n,n), term2[i,l] = sum_j a[j,i]*b[p[j],l]
    
    C = term1 + term2
    return C


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
            # Build initial surrogate from spectral solution
            surrogate = _build_surrogate(a, b, p_spec)
            p_ls, cost_ls = _local_search(a, b, p_spec.copy(), deadline - 5, surrogate=surrogate)
            _update_best(p_ls, cost_ls)
            # Update surrogate with improved solution
            surrogate = _build_surrogate(a, b, p_ls)

    # 2. Benders-style Master: solve assignment with surrogate
    if time.time() < deadline - 10:
        # Use current surrogate for master problem
        row_ind, col_ind = linear_sum_assignment(surrogate)
        p_master = np.empty(n, dtype=np.int64)
        p_master[row_ind] = col_ind
        cost_master = _cost(a, b, p_master)
        _update_best(p_master, cost_master)

        if time.time() < deadline - 5:
            p_ls, cost_ls = _local_search(a, b, p_master.copy(), deadline - 5, surrogate=surrogate)
            _update_best(p_ls, cost_ls)
            # Generate a "cut" by updating surrogate with this solution
            surrogate_new = _build_surrogate(a, b, p_ls)
            # Blend: keep some of old surrogate structure (for stability)
            surrogate = 0.5 * surrogate + 0.5 * surrogate_new

    # 3. Iterated Local Search (ILS) with Benders-guided perturbation
    if best_p is not None and time.time() < deadline - 5:
        current_p = best_p.copy()
        current_cost = best_cost
        rng = np.random.default_rng(42)
        
        # Build surrogate from best solution
        surrogate = _build_surrogate(a, b, current_p)
        
        while time.time() < deadline - 2:
            # Benders-style perturbation: use surrogate to find a "good" reassignment
            # Solve master with perturbed surrogate
            if time.time() < deadline - 3:
                # Add noise to surrogate to escape local minima
                noise_scale = 0.1 * (np.max(surrogate) - np.min(surrogate) + 1e-9)
                surrogate_perturbed = surrogate + noise_scale * rng.standard_normal((n, n))
                row_ind, col_ind = linear_sum_assignment(surrogate_perturbed)
                p_perturbed = np.empty(n, dtype=np.int64)
                p_perturbed[row_ind] = col_ind
            else:
                # Fallback: random swaps
                p_perturbed = current_p.copy()
                n_swaps = rng.integers(3, 8)
                for _ in range(n_swaps):
                    i, j = rng.integers(0, n, 2)
                    if i != j:
                        p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
            
            # Local search on perturbed solution, guided by surrogate
            p_ls, cost_ls = _local_search(a, b, p_perturbed, deadline - 2, surrogate=surrogate)
            
            # Accept if better
            if cost_ls < current_cost:
                current_p = p_ls
                current_cost = cost_ls
                _update_best(current_p, current_cost)
                # Update surrogate with improved solution
                surrogate = _build_surrogate(a, b, current_p)
            else:
                # Occasionally update surrogate even if not improved, to diversify
                if rng.random() < 0.3:
                    surrogate = _build_surrogate(a, b, p_ls)

    if best_p is None:
        best_p = np.arange(n)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
