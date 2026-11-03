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
    """Iterated local search using 2-opt style moves."""
    n = len(a)
    p = np.array(p, dtype=np.int64)
    current_cost = _evaluate(a, b, p)
    
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    
    for iteration in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Randomly pick two indices to swap
        i, j = np.random.default_rng().choice(n, size=2, replace=False)
        
        if i == j:
            continue
        
        # Compute the cost change for swapping p[i] and p[j]
        # delta = new_cost - old_cost
        # old: a[i,i]*b[p[i],p[i]] + a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]] + a[j,j]*b[p[j],p[j]]
        #      + sum_{k != i,j} a[k,i]*b[p[k],p[i]] + sum_{k != i,j} a[i,k]*b[p[i],p[k]]
        #      + sum_{k != i,j} a[k,j]*b[p[k],p[j]] + sum_{k != i,j} a[j,k]*b[p[j],p[k]]
        # new: a[i,i]*b[p[j],p[j]] + a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[i],p[j]] + a[j,j]*b[p[i],p[i]]
        #      + sum_{k != i,j} a[k,i]*b[p[k],p[j]] + sum_{k != i,j} a[i,k]*b[p[j],p[k]]
        #      + sum_{k != i,j} a[k,j]*b[p[k],p[i]] + sum_{k != i,j} a[j,k]*b[p[i],p[k]]
        
        # delta = (a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[i],p[j]] + a[i,i]*b[p[j],p[j]] + a[j,j]*b[p[i],p[i]])
        #       - (a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]] + a[i,i]*b[p[i],p[i]] + a[j,j]*b[p[j],p[j]])
        #       + sum_{k != i,j} [a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]]) + a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]])
        #                         + a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]]) + a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]])]
        
        pi = p[i]
        pj = p[j]
        
        # The direct terms
        delta_direct = (a[i, j] * b[pj, pi] + a[j, i] * b[pi, pj] + a[i, i] * b[pj, pj] + a[j, j] * b[pi, pi]) \
                     - (a[i, j] * b[pi, pj] + a[j, i] * b[pj, pi] + a[i, i] * b[pi, pi] + a[j, j] * b[pj, pj])
        
        # The indirect terms: for all k != i, j
        # We need: sum_{k != i,j} a[k,i] * (b[p[k], pj] - b[p[k], pi])
        #         + sum_{k != i,j} a[i,k] * (b[pj, p[k]] - b[pi, p[k]])
        #         + sum_{k != i,j} a[k,j] * (b[p[k], pi] - b[p[k], pj])
        #         + sum_{k != i,j} a[j,k] * (b[pi, p[k]] - b[pj, p[k]])
        
        # Note that b[p[k], pj] - b[p[k], pi] = b[:, pj][p[k]] - b[:, pi][p[k]]
        # Let's precompute column differences
        col_diff_ij = b[:, pj] - b[:, pi]  # shape (n,)
        row_diff_ij = b[pi, :] - b[pj, :]  # shape (n,)
        
        # For k != i, j:
        # sum_k a[k,i] * col_diff_ij[p[k]]  (for k != i, j)
        # sum_k a[i,k] * row_diff_ij[p[k]]  (for k != i, j)
        # sum_k a[k,j] * (-col_diff_ij[p[k]])  (for k != i, j)  [since b[p[k],pi]-b[p[k],pj] = -col_diff_ij[p[k]]]
        # sum_k a[j,k] * (-row_diff_ij[p[k]])  (for k != i, j)  [since b[pi,p[k]]-b[pj,p[k]] = row_diff_ij[p[k]]... wait]
        
        # Let me redo this more carefully.
        # Term 1: sum_{k != i,j} a[k,i] * (b[p[k], pj] - b[p[k], pi])
        #        = sum_{k != i,j} a[k,i] * col_diff_ij[p[k]]
        
        # Term 2: sum_{k != i,j} a[i,k] * (b[pj, p[k]] - b[pi, p[k]])
        #        = sum_{k != i,j} a[i,k] * (-row_diff_ij[p[k]])  [since row_diff_ij = b[pi,:]-b[pj,:], so b[pj,p[k]]-b[pi,p[k]] = -row_diff_ij[p[k]]]
        
        # Term 3: sum_{k != i,j} a[k,j] * (b[p[k], pi] - b[p[k], pj])
        #        = sum_{k != i,j} a[k,j] * (-col_diff_ij[p[k]])
        
        # Term 4: sum_{k != i,j} a[j,k] * (b[pi, p[k]] - b[pj, p[k]])
        #        = sum_{k != i,j} a[j,k] * row_diff_ij[p[k]]
        
        # So:
        # indirect = sum_{k != i,j} [a[k,i] - a[k,j]] * col_diff_ij[p[k]]
        #          + sum_{k != i,j} [a[j,k] - a[i,k]] * row_diff_ij[p[k]]
        
        # We need to exclude k=i and k=j
        # Let's compute the full sum and subtract the i and j terms
        
        # Full sum for col_diff:
        # sum_k (a[:,i] - a[:,j]) * col_diff_ij[p[k]]
        # = (a[:,i] - a[:,j]) @ col_diff_ij[p]
        
        # But we need to exclude k=i and k=j
        # When k=i: (a[i,i] - a[i,j]) * col_diff_ij[p[i]] = (a[i,i] - a[i,j]) * col_diff_ij[pi]
        # When k=j: (a[j,i] - a[j,j]) * col_diff_ij[p[j]] = (a[j,i] - a[j,j]) * col_diff_ij[pj]
        
        col_diff_at_p = col_diff_ij[p]  # shape (n,)
        row_diff_at_p = row_diff_ij[p]  # shape (n,)
        
        a_col_diff = a[:, i] - a[:, j]  # shape (n,)
        a_row_diff = a[j, :] - a[i, :]  # shape (n,)
        
        full_col_sum = np.dot(a_col_diff, col_diff_at_p)
        full_row_sum = np.dot(a_row_diff, row_diff_at_p)
        
        # Subtract k=i and k=j terms
        # For col: k=i term is a_col_diff[i] * col_diff_at_p[i] = (a[i,i]-a[i,j]) * col_diff_at_p[i]
        #          k=j term is a_col_diff[j] * col_diff_at_p[j] = (a[j,i]-a[j,j]) * col_diff_at_p[j]
        col_excl = full_col_sum - a_col_diff[i] * col_diff_at_p[i] - a_col_diff[j] * col_diff_at_p[j]
        
        # For row: k=i term is a_row_diff[i] * row_diff_at_p[i] = (a[j,i]-a[i,i]) * row_diff_at_p[i]
        #          k=j term is a_row_diff[j] * row_diff_at_p[j] = (a[j,j]-a[i,j]) * row_diff_at_p[j]
        row_excl = full_row_sum - a_row_diff[i] * row_diff_at_p[i] - a_row_diff[j] * row_diff_at_p[j]
        
        delta = delta_direct + col_excl + row_excl
        
        if delta < 0:
            # Accept the swap
            p[i], p[j] = p[j], p[i]
            current_cost += delta
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
        
        p_ls, cost_ls = _local_search(a, b, p_init, max_iters=3000, time_limit=remaining_time - 2)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    # If no local search was run (shouldn't happen), use spectral init
    if best_p is None:
        best_p = _spectral_init(a, b, k=10)
    
    return best_p
# EVOLVE-BLOCK-END
