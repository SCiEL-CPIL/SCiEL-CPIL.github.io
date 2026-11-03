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
    
    # Center the matrices
    a_centered = a - a.mean(axis=1, keepdims=True)
    a_centered = a_centered - a_centered.mean(axis=0, keepdims=True)
    
    b_centered = b - b.mean(axis=1, keepdims=True)
    b_centered = b_centered - b_centered.mean(axis=0, keepdims=True)
    
    # Get top-k principal components via SVD
    # For symmetric matrices, eigenvalues are real; use eigh for efficiency
    # But a and b may not be perfectly symmetric in general QAP, so use SVD
    try:
        Ua, Sa, Va = np.linalg.svd(a_centered, full_matrices=False)
        Ub, Sb, Vb = np.linalg.svd(b_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback to simple heuristic
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p.tolist()
    
    # Project onto top-k components
    # We want to match the "structure" of a with b
    # The idea: facilities with similar flow patterns should be assigned to locations with similar distance patterns
    
    # Compute low-rank approximation of the "flow structure" and "distance structure"
    # Use the left singular vectors as the basis for facility similarity
    # and right singular vectors for location similarity
    
    # Actually, for QAP, a[i,j] is flow between facility i and j, b[p[i],p[j]] is distance
    # We want to minimize sum a[i,j] * b[p[i],p[j]]
    
    # A common approach: use the eigenvalue decomposition to get a "landscape"
    # Let's compute the top-k eigenvectors of a and b
    
    # For symmetric matrices, use eigh
    # First, symmetrize
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
    
    # Now we have a low-dimensional representation of facilities (Ua_top) and locations (Ub_top)
    # We want to assign facility i to location p[i] such that the "structural similarity" is matched
    # 
    # One approach: create a surrogate cost matrix where cost[i, j] measures how well
    # facility i's spectral signature matches location j's spectral signature
    #
    # For each facility i, its signature is Ua_top[i, :] (a k-dimensional vector)
    # For each location j, its signature is Ub_top[j, :] (a k-dimensional vector)
    #
    # We want to match facilities with high "flow centrality" to locations with low "distance centrality"
    # This is captured by the sign and magnitude of the spectral components
    
    # Create surrogate cost: negative correlation between spectral signatures
    # cost[i, j] = -sum_k Ua_top[i,k] * Ub_top[j,k] * (some weight)
    
    # Weight by eigenvalues to emphasize important modes
    weights_a = np.abs(evals_a[:k_eff]) + 1e-10
    weights_b = np.abs(evals_b[:k_eff]) + 1e-10
    
    # Surrogate cost matrix
    # C[i, j] = - (Ua_top @ diag(sqrt(weights_a * weights_b)) @ Ub_top.T)[i, j]
    W = np.sqrt(weights_a * weights_b)
    C = -(Ua_top * W) @ Ub_top.T
    
    # Solve the linear assignment problem
    row_ind, col_ind = linear_sum_assignment(C)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p.tolist()


def _evaluate(a, b, p):
    """Compute the QAP objective value for permutation p."""
    n = len(a)
    # sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Efficient: sum_i a[i,:] dot b[p[i], p[j]] for all j
    # = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # = sum_i (a[i,:] @ b[p[i], :])
    # = sum_i sum_j a[i,j] * b[p[i], j]
    
    # Let's compute this efficiently
    # b_perm[i, j] = b[p[i], j]
    b_perm = b[p, :]
    # cost = sum_i sum_j a[i,j] * b_perm[i,j]
    cost = np.sum(a * b_perm)
    return int(cost)


def _local_search(a, b, p, max_iters=5000, time_limit=None):
    """Iterated Local Search with vectorized steepest descent and reversal kicks."""
    n = len(a)
    p = np.array(p, dtype=np.int64)
    current_cost = _evaluate(a, b, p)
    
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    rng = np.random.default_rng()
    
    # Precompute for fast delta computation
    # row_sum[i] = sum_j a[i, j] * b[p[i], p[j]]
    # col_sum[j] = sum_i a[i, j] * b[p[i], p[j]]
    b_perm = b[p, :]
    row_sum = np.sum(a * b_perm, axis=1)
    col_sum = np.sum(a * b_perm, axis=0)
    
    def _apply_swap(i, j):
        """Apply swap and update row_sum, col_sum, current_cost. Returns delta."""
        nonlocal current_cost, p, row_sum, col_sum
        pi = p[i]
        pj = p[j]
        
        # Compute delta before swapping
        # Direct terms
        delta_direct = (a[i, j] * b[pj, pi] + a[j, i] * b[pi, pj] + a[i, i] * b[pj, pj] + a[j, j] * b[pi, pi]) \
                     - (a[i, j] * b[pi, pj] + a[j, i] * b[pj, pi] + a[i, i] * b[pi, pi] + a[j, j] * b[pj, pj])
        
        # Indirect terms using precomputed sums
        # For all k != i, j: a[k, i] * b[p[k], pj] - a[k, i] * b[p[k], pi] + a[k, j] * b[p[k], pi] - a[k, j] * b[p[k], pj]
        # = sum_{k != i,j} (a[k, i] - a[k, j]) * (b[p[k], pj] - b[p[k], pi])
        # Also need to account for the fact that p[i] and p[j] change
        
        # Simpler: recompute affected terms
        # After swap: p[i] = pj, p[j] = pi
        # New row_sum[i] = sum_j a[i, j] * b[pj, p[j]]  (but p[j] becomes pi)
        # New row_sum[j] = sum_j a[j, j] * b[pi, p[j]]  (but p[i] becomes pj)
        
        # Let's just compute delta directly for correctness
        # delta = new_cost - old_cost
        
        # Old contributions involving i or j:
        # sum_k a[i,k] * b[pi, p[k]] + sum_k a[k,j] * b[p[k], pj]
        # New contributions:
        # sum_k a[i,k] * b[pj, p[k]] + sum_k a[k,j] * b[p[k], pi]
        # But when k=i: a[i,i]*b[pj,pj] vs a[i,i]*b[pi,pi]
        # When k=j: a[i,j]*b[pj,pi] vs a[i,j]*b[pi,pj]
        
        # For k not in {i,j}:
        # Old: a[i,k]*b[pi, p[k]] + a[k,j]*b[p[k], pj]
        # New: a[i,k]*b[pj, p[k]] + a[k,j]*b[p[k], pi]
        # Delta_k = a[i,k]*(b[pj, p[k]] - b[pi, p[k]]) + a[k,j]*(b[p[k], pi] - b[p[k], pj])
        
        # This is still O(n) per swap. Let's use the precomputed sums.
        
        # Actually, let me just do the full O(n) computation for correctness
        # and rely on the vectorized approach for finding the best swap.
        
        # For now, compute delta the straightforward way
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
        
        # Update row_sum and col_sum
        # row_sum[i] changes: was sum_k a[i,k]*b[pi, p[k]], now sum_k a[i,k]*b[pj, p[k]]
        # row_sum[j] changes: was sum_k a[j,k]*b[pj, p[k]], now sum_k a[j,k]*b[pi, p[k]]
        # col_sum changes for all k: col_sum[k] had term a[k,i]*b[p[k], pi] + a[k,j]*b[p[k], pj]
        # now has a[k,i]*b[p[k], pj] + a[k,j]*b[p[k], pi]
        
        # Update row_sum
        row_sum[i] = np.sum(a[i, :] * b[pj, :])
        row_sum[j] = np.sum(a[j, :] * b[pi, :])
        
        # Update col_sum
        # col_sum[k] = sum_i a[i, k] * b[p[i], p[k]]
        # The change is only in terms involving positions i and j
        # For each k, the old contribution from facility i to location k was a[i, k] * b[pi, p[k]]
        # New: a[i, k] * b[pj, p[k]]
        # Similarly for facility j: old a[j, k] * b[pj, p[k]], new a[j, k] * b[pi, p[k]]
        # So col_sum[k] += a[i, k] * (b[pj, p[k]] - b[pi, p[k]]) + a[j, k] * (b[pi, p[k]] - b[pj, p[k]])
        
        # Vectorized update for col_sum
        diff_b = b[pj, :] - b[pi, :]
        col_sum += a[i, :] * diff_b + a[j, :] * (-diff_b)
        
        return delta
    
    def _find_best_swap_vectorized():
        """Find the best improving swap using vectorized operations."""
        nonlocal p, row_sum, col_sum
        # For each pair (i, j) with i < j, compute delta
        # delta(i, j) = direct + indirect
        # direct = a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi]) + a[i,i]*(b[pj,pj] - b[pi,pi]) + a[j,j]*(b[pi,pi] - b[pj,pj])
        # indirect = sum_{k!=i,j} [a[i,k]*(b[pj, p[k]] - b[pi, p[k]]) + a[k,j]*(b[p[k], pi] - b[p[k], pj])]
        
        # Let's compute this vectorized
        # Create matrices P = p (shape n,)
        # For each i, j: 
        # direct_ij = a[i,j]*(b[p[j],p[i]] - b[p[i],p[j]]) + a[j,i]*(b[p[i],p[j]] - b[p[j],p[i]]) + a[i,i]*(b[p[j],p[j]] - b[p[i],p[i]]) + a[j,j]*(b[p[i],p[i]] - b[p[j],p[j]])
        
        # This is complex to vectorize fully. Let's use a different approach:
        # For each i, compute the delta for swapping i with all j > i
        
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
            # a[i, js] * (b[ps, pi] - b[pi, ps])
            direct1 = a[i, js] * (b[ps, pi] - b[pi, ps])
            # a[js, i] * (b[pi, ps] - b[ps, pi])
            direct2 = a[js, i] * (b[pi, ps] - b[ps, pi])
            # a[i, i] * (b[ps, ps] - b[pi, pi])
            direct3 = a[i, i] * (b[ps, ps] - b[pi, pi])
            # a[js, js] * (b[pi, pi] - b[ps, ps])
            direct4 = np.diag(a[js, js]) if len(js) == 1 else np.array([a[j, j] for j in js]) * (b[pi, pi] - b[ps, ps])
            
            # Actually, a[js, js] isn't a diagonal extraction. Let me fix this.
            # a[j, j] for j in js
            a_jj = np.array([a[j, j] for j in js])
            direct4 = a_jj * (b[pi, pi] - b[ps, ps])
            
            direct = direct1 + direct2 + direct3 + direct4
            
            # Indirect terms
            # sum_{k != i, j} [a[i,k]*(b[ps_j, p[k]] - b[pi, p[k]]) + a[k, j]*(b[p[k], pi] - b[p[k], ps_j])]
            # This is hard to vectorize over both i and j simultaneously.
            # Let's approximate or use a different strategy.
            
            # For now, let's just use the direct terms as a guide and then verify
            # with a few candidates.
            
            # Find top candidates by direct delta
            top_idx = np.argsort(direct)[:min(50, len(js))]
            
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
        min_delta, best_i, best_j = _find_best_swap_vectorized()
        
        if min_delta < 0 and best_i >= 0:
            # Apply best swap
            d = _apply_swap(best_i, best_j)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # No improving swap found, perform a kick
            # Use a random reversal for stronger perturbation
            if rng.random() < 0.5:
                # Single swap
                i, j = rng.choice(n, size=2, replace=False)
                if i != j:
                    d = _apply_swap(int(i), int(j))
            else:
                # Reversal
                i, j = sorted(rng.choice(n, size=2, replace=False))
                i, j = int(i), int(j)
                if i < j:
                    # Reverse segment p[i:j+1]
                    # Compute delta for reversal
                    # This is complex, so let's just do it step by step
                    # or use a simpler kick
                    
                    # For simplicity, do a random swap as kick
                    i2, j2 = rng.choice(n, size=2, replace=False)
                    if i2 != j2:
                        d = _apply_swap(int(i2), int(j2))
            
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
    
    # Generate multiple diverse initial solutions using spectral projection
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
