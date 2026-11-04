"""Multi-Start Iterated Local Search with spectral-projection initialization for QAP."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def _spectral_init(a, b, k=10, rng=None):
    """Generate an initial permutation using spectral decomposition of a and b."""
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(a)
    
    # Symmetrize
    a_sym = 0.5 * (a + a.T)
    b_sym = 0.5 * (b + b.T)
    
    try:
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
    except np.linalg.LinAlgError:
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p.tolist()
    
    # Sort by eigenvalue descending
    idx_a = np.argsort(-evals_a)
    idx_b = np.argsort(-evals_b)
    evals_a = evals_a[idx_a]
    evecs_a = evecs_a[:, idx_a]
    evals_b = evals_b[idx_b]
    evecs_b = evecs_b[:, idx_b]
    
    # Take top-k
    k_eff = min(k, n)
    Ua_top = evecs_a[:, :k_eff]
    Ub_top = evecs_b[:, :k_eff]
    
    # Weight by eigenvalues
    weights_a = np.abs(evals_a[:k_eff]) + 1e-10
    weights_b = np.abs(evals_b[:k_eff]) + 1e-10
    
    # Surrogate cost matrix
    W = np.sqrt(weights_a * weights_b)
    C = -(Ua_top * W) @ Ub_top.T
    
    # Solve the linear assignment problem
    row_ind, col_ind = linear_sum_assignment(C)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p.tolist()


def _evaluate(a, b, p):
    """Compute the QAP objective value for permutation p."""
    b_perm = b[p, :]
    cost = np.sum(a * b_perm)
    return int(cost)


def _local_search(a, b, p, max_iters=5000, time_limit=None):
    """Iterated Local Search with vectorized best-swap and reversal kicks."""
    n = len(a)
    p = np.array(p, dtype=np.int64)
    current_cost = _evaluate(a, b, p)
    
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    rng = np.random.default_rng()
    
    def _apply_swap(i, j):
        """Apply swap and update current_cost. Returns delta."""
        nonlocal current_cost, p
        pi = p[i]
        pj = p[j]
        
        # Compute delta
        delta = 0
        # Direct
        delta += a[i, j] * (b[pj, pi] - b[pi, pj])
        delta += a[j, i] * (b[pi, pj] - b[pj, pi])
        delta += a[i, i] * (b[pj, pj] - b[pi, pi])
        delta += a[j, j] * (b[pi, pi] - b[pj, pj])
        
        # Indirect: for all k != i, j
        for k in range(n):
            if k == i or k == j:
                continue
            pk = p[k]
            delta += a[i, k] * (b[pj, pk] - b[pi, pk])
            delta += a[k, j] * (b[pk, pi] - b[pk, pj])
        
        # Apply the swap
        p[i], p[j] = pj, pi
        current_cost += delta
        
        return delta
    
    def _find_best_swap():
        """Find the best improving swap using vectorized operations."""
        nonlocal p
        min_delta = 0
        best_i, best_j = -1, -1
        
        for i in range(n):
            pi = p[i]
            # For all j > i
            js = np.arange(i + 1, n)
            if len(js) == 0:
                continue
            ps = p[js]
            
            # Direct terms
            direct1 = a[i, js] * (b[ps, pi] - b[pi, ps])
            direct2 = a[js, i] * (b[pi, ps] - b[ps, pi])
            direct3 = a[i, i] * (b[ps, ps] - b[pi, pi])
            a_jj = np.array([a[j, j] for j in js])
            direct4 = a_jj * (b[pi, pi] - b[ps, ps])
            
            direct = direct1 + direct2 + direct3 + direct4
            
            # Find top candidates by direct delta
            top_idx = np.argsort(direct)[:min(30, len(js))]
            
            for idx in top_idx:
                j = js[idx]
                pj = p[j]
                
                # Compute full delta
                delta = direct[idx]
                
                # Indirect
                for k in range(n):
                    if k == i or k == j:
                        continue
                    pk = p[k]
                    delta += a[i, k] * (b[pj, pk] - b[pi, pk])
                    delta += a[k, j] * (b[pk, pi] - b[pk, pj])
                
                if delta < min_delta:
                    min_delta = delta
                    best_i = i
                    best_j = j
        
        return min_delta, best_i, best_j
    
    for iteration in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Find best swap
        min_delta, best_i, best_j = _find_best_swap()
        
        if min_delta < 0 and best_i >= 0:
            # Apply best swap
            d = _apply_swap(best_i, best_j)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # No improving swap found, perform a kick
            if rng.random() < 0.5:
                # Single swap
                i, j = rng.choice(n, size=2, replace=False)
                if i != j:
                    d = _apply_swap(int(i), int(j))
            else:
                # Reversal kick: reverse a random segment
                i, j = sorted(rng.choice(n, size=2, replace=False))
                i, j = int(i), int(j)
                if i < j:
                    # Reverse segment p[i:j+1]
                    p[i:j+1] = p[i:j+1][::-1]
                    # Recompute cost
                    current_cost = _evaluate(a, b, p)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p.tolist(), best_cost


def solve(a, b):
    """Multi-Start Iterated Local Search for QAP."""
    n = len(a)
    
    # Convert to numpy arrays if needed
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    start_time = time.time()
    time_limit = 115  # Leave some margin for the 120s limit
    
    # Generate multiple diverse initial solutions
    best_p = None
    best_cost = np.inf
    
    num_starts = 5
    
    for start_idx in range(num_starts):
        if time.time() - start_time > time_limit - 30:
            break
        
        # Generate initial permutation
        if start_idx == 0:
            p_init = _spectral_init(a, b, k=10)
        elif start_idx == 1:
            p_init = _spectral_init(a, b, k=5)
        elif start_idx == 2:
            # Rank-matching heuristic
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p_arr = np.empty(n, dtype=np.int64)
            p_arr[heavy] = central
            p_init = p_arr.tolist()
        elif start_idx == 3:
            # Random
            p_init = list(np.random.default_rng(start_idx).permutation(n))
        else:
            # Another spectral with different k
            p_init = _spectral_init(a, b, k=15)
        
        # Run local search
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 5:
            break
        
        p_ls, cost_ls = _local_search(a, b, p_init, max_iters=5000, time_limit=remaining_time - 2)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    # If no local search was run (shouldn't happen), use spectral init
    if best_p is None:
        best_p = _spectral_init(a, b, k=10)
    
    return best_p
# EVOLVE-BLOCK-END
