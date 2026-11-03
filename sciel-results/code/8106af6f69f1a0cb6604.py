# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute the QAP cost for a given permutation p."""
    # a is n x n, b is n x n, p is a permutation of 0..n-1
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    return np.sum(a * b[np.ix_(p, p)])


def _spectral_init(a, b):
    """Generate an initial permutation using spectral embedding."""
    n = a.shape[0]
    
    # Build the linearization matrix M
    # M = diag(a @ 1) + a * b  (element-wise product, which is a common QAP linearization)
    # Actually, a better spectral approach: 
    # The QAP can be approximated by embedding facilities into a space where distances match.
    # One common approach: use the matrix C = a * b (Hadamard product) and find eigenvectors.
    
    # Let's use the approach: M = a * b (element-wise)
    # Then find the leading eigenvector and use it to order facilities.
    # But this gives a single vector. We need a permutation.
    
    # Alternative: use the matrix L = a * b, compute its eigen-decomposition,
    # take the first few eigenvectors to embed facilities in R^k, then use a greedy
    # nearest-neighbor or sorting approach to generate a permutation.
    
    # Actually, a simpler and effective spectral initialization:
    # 1. Compute M = a * b (element-wise product)
    # 2. Symmetrize: M_sym = (M + M.T) / 2
    # 3. Find the leading eigenvector
    # 4. Use the eigenvalues to create a "score" for each facility-location pair
    
    # Let's try a different approach: 
    # The QAP objective is sum_{i,j} a_{ij} b_{pi,pj}.
    # If we fix p, this is sum over all pairs.
    # A spectral approach: consider the matrix A = a, and we want to assign p such that
    # the "aligned" matrix a * b[p,p] has large trace (if we were maximizing).
    # Since we're minimizing, we want small.
    
    # Practical spectral init: 
    # 1. Compute row sums of a and b
    # 2. Use a combination of spectral and rank-based initialization
    
    # Let me try: use the eigenvectors of (a * b) to get a low-dimensional embedding
    M = a * b  # Hadamard product
    M_sym = (M + M.T) / 2.0
    
    # Get leading eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(M_sym)
        # Sort by eigenvalue magnitude (descending)
        idx = np.argsort(-np.abs(eigvals))
        eigvecs = eigvecs[:, idx]
        
        # Use the first few eigenvectors to embed facilities
        k = min(4, n)
        embed = eigvecs[:, :k]
        
        # Project onto a 1D space using the first eigenvector
        scores = embed[:, 0]
        # Sort facilities by their spectral score
        order = np.argsort(scores, kind='stable')
        
        # Create initial permutation: facility i goes to location order[i]
        p = np.empty(n, dtype=np.int64)
        p[order] = np.arange(n)
        return p
    except Exception:
        # Fallback to rank-based
        heavy = np.argsort(-a.sum(axis=1), kind='stable')
        central = np.argsort(b.sum(axis=1), kind='stable')
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p


def _greedy_init(a, b):
    """Generate an initial permutation using greedy rank matching."""
    n = a.shape[0]
    heavy = np.argsort(-a.sum(axis=1), kind='stable')
    central = np.argsort(b.sum(axis=1), kind='stable')
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _local_search_2opt(a, b, p, time_limit=10.0):
    """
    Vectorized 2-opt local search with O(n^2) swap evaluation.
    Uses the R-matrix formulation for efficient delta computation.
    """
    n = a.shape[0]
    p = p.copy().astype(np.int64)
    start_time = time.time()
    
    # Precompute R[i, l] = sum_j a[i,j] * b[l, p[j]]
    # This is n x n
    # R[i, :] = a[i, :] @ b[:, p] ... wait, let me think
    # R[i, l] = sum_j a[i,j] * b[l, p[j]]
    # = (a[i, :] * b[:, p[:, 0]] ... no
    # For fixed i: R[i, l] = sum_j a[i,j] * b[l, p[j]]
    # Let B_p = b[:, p]  (b with columns permuted by p)
    # Then R[i, l] = sum_j a[i,j] * B_p[j, l]  ... no, B_p[j, l] = b[l, p[j]]
    # So R = a @ B_p.T ? Let's check:
    # (a @ B_p.T)[i, l] = sum_j a[i,j] * B_p.T[j, l] = sum_j a[i,j] * B_p[l, j] = sum_j a[i,j] * b[p[l], j]
    # That's not quite right. We want b[l, p[j]], not b[p[l], j].
    # 
    # Let me redefine: B_perm = b[np.ix_(np.arange(n), p)]  # B_perm[l, j] = b[l, p[j]]
    # Then R[i, l] = sum_j a[i,j] * B_perm[l, j] = (a @ B_perm.T)[i, l]
    # Yes! So R = a @ B_perm.T where B_perm[l, j] = b[l, p[j]]
    
    def compute_R(p):
        B_perm = b[np.ix_(np.arange(n), p)]  # B_perm[l, j] = b[l, p[j]]
        return a @ B_perm.T  # R[i, l] = sum_j a[i,j] * b[l, p[j]]
    
    R = compute_R(p)
    
    # Current cost
    current_cost = _qap_cost(a, b, p)
    
    # Precompute C[i,k] = a[k,i] - a[i,k] + a[i,i] - a[k,k]
    # This is used in the delta formula
    # C[i,k] = a[k,i] - a[i,k] + a[i,i] - a[k,k]
    C = a.T - a + np.diag(np.diag(a))[:, None] - np.diag(np.diag(a))[None, :]
    # Wait, let me recompute:
    # C[i,k] = a[k,i] - a[i,k] + a[i,i] - a[k,k]
    # a[k,i] - a[i,k] = (a.T - a)[i, k]
    # a[i,i] - a[k,k] = diag(a)[i] - diag(a)[k]
    diag_a = np.diag(a)
    C = (a.T - a) + diag_a[:, None] - diag_a[None, :]
    
    # B_diag = diag(b)
    B_diag = np.diag(b)
    
    # Precompute B_diff[v, u] = b[v, v] - b[u, u] for use in delta
    # Actually, we need B[p[k], p[k]] - B[p[i], p[i]] = B_diag[p[k]] - B_diag[p[i]]
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Find the best swap (i, k) with i < k
        # Delta[i,k] = R[i, p[k]] - R[i, p[i]] + R[k, p[i]] - R[k, p[k]] + C[i,k] * (B_diag[p[k]] - B_diag[p[i]])
        
        # For all i, k:
        # u = p[i], v = p[k]
        # Delta[i,k] = R[i, v] - R[i, u] + R[k, u] - R[k, v] + C[i,k] * (B_diag[v] - B_diag[u])
        
        # Vectorize:
        # R[i, p[k]] for all i,k: R[np.arange(n)[:,None], p[None,:]]  -> n x n
        # R[i, p[i]] for all i: R[np.arange(n), p]  -> n
        # R[k, p[i]] for all i,k: R[p[None,:], np.arange(n)[:,None]]  -> n x n (transposed)
        # R[k, p[k]] for all k: R[np.arange(n), p]  -> n
        
        idx = np.arange(n)
        
        # R[i, p[k]]: shape (n, n), entry (i,k) = R[i, p[k]]
        Ripk = R[idx[:, None], p[None, :]]  # R[i, p[k]]
        
        # R[i, p[i]]: shape (n,), entry i = R[i, p[i]]
        Rii = R[idx, p]
        
        # R[k, p[i]]: entry (i,k) = R[k, p[i]] = R[p[i], k]... wait
        # We want R[k, p[i]] where k is the row and p[i] is the column.
        # Entry (i,k) in our matrix should be R[k, p[i]]
        # So: R[p[None,:], idx[:, None]] gives entry (i,k) = R[p[i], k] ... no
        # R[p[i], k] is not what we want. We want R[k, p[i]].
        # R[k, p[i]]: row k, column p[i].
        # For entry (i,k): row = k, col = p[i]
        # Rkpi = R[p[None, :], idx[:, None]]  # entry (i,k) = R[p[i], k] -- WRONG
        # We need: R[idx[None, :], p[:, None]]  # entry (i,k) = R[k, p[i]]? No
        # R[idx[None, :], p[:, None]]: entry (i,k) = R[idx[i], p[k]] = R[i, p[k]] -- that's Ripk again
        
        # Let me be more careful:
        # We want a matrix D where D[i,k] = R[k, p[i]]
        # D[i,k] = R[k, p[i]]
        # This is: rows indexed by k, columns indexed by p[i]
        # D = R.T @ ... no, let's just use advanced indexing
        # D[i,k] = R[k, p[i]]
        # D = R[p[None, :] + 0, ...] hmm
        
        # D[i,k] = R[k, p[i]]
        # For fixed i, as k varies: D[i,:] = R[:, p[i]]
        # So D = np.array([R[:, p[i]] for i in range(n)])  -> shape (n, n), entry (i,k) = R[k, p[i]]
        # Vectorized: D = R.T[:, p]  ? Let's check:
        # R.T has shape (n,n), R.T[l, m] = R[m, l]
        # R.T[:, p] has shape (n, n), entry (l, j) = R.T[l, p[j]] = R[p[j], l]
        # We want D[i,k] = R[k, p[i]], so l=k, p[j]=p[i] -> j=i
        # R.T[:, p][k, i] = R[p[i], k] -- that's R[p[i], k], not R[k, p[i]]
        
        # OK let me just do it explicitly with broadcasting:
        # D[i,k] = R[k, p[i]]
        # D = R[idx[None, :], p[:, None]]  # entry (i,k) = R[idx[i], p[k]] = R[i, p[k]] -- no
        # I keep getting confused. Let me use:
        # D[i,k] = R[k, p[i]]
        # Let rows = k (0..n-1), cols = i (0..n-1)
        # D.T[i,k] = D[k,i] = R[i, p[k]]... no
        
        # Let's just compute it as:
        # D[i,k] = R[k, p[i]]
        # Create an array where for each i, we take column p[i] of R, which gives R[:, p[i]] of shape (n,)
        # Then D = np.stack([R[:, p[i]] for i in range(n)], axis=1)  # shape (n, n), D[k, i] = R[k, p[i]]
        # So D[k,i] = R[k, p[i]], meaning D.T[i,k] = R[k, p[i]]
        # So we want D_T = D.T where D_T[i,k] = R[k, p[i]]
        # D = np.stack([R[:, p[i]] for i in range(n)], axis=1)
        # This is O(n^2) but done in a loop. Let's vectorize:
        # R[:, p[i]] for all i: this is R[:, p] with p of shape (n,)
        # R[:, p] has shape (n, n), entry (l, j) = R[l, p[j]]
        # So R[:, p][k, i] = R[k, p[i]] -- YES!
        # So D[i,k] = R[k, p[i]] = R[:, p][k, i]
        # We want a matrix where entry (i,k) = R[:, p][k, i]
        # That's (R[:, p]).T
        Rkpi = (R[:, p]).T  # Rkpi[i,k] = R[k, p[i]]
        
        # R[k, p[k]]: entry (i,k) should be R[k, p[k]]
        # R[k, p[k]] for all k: R[idx, p] = Rii (already computed)
        # We need a matrix where entry (i,k) = R[k, p[k]] = Rii[k]
        # So Rkk = np.tile(Rii[None, :], (n, 1))  # shape (n,n), entry (i,k) = Rii[k]
        Rkk = np.tile(Rii[None, :], (n, 1))
        
        # C[i,k] is already shape (n,n)
        
        # B_diag[p[k]] - B_diag[p[i]]: entry (i,k) = B_diag[p[k]] - B_diag[p[i]]
        Bd_diff = B_diag[p[None, :]] - B_diag[p[:, None]]  # entry (i,k) = B_diag[p[k]] - B_diag[p[i]]
        
        # Delta[i,k] = Ripk - Rii[:, None] + Rkpi - Rkk + C * Bd_diff
        # Rii[:, None] has shape (n,1), broadcast to (n,n), entry (i,k) = Rii[i] = R[i, p[i]]
        Delta = Ripk - Rii[:, None] + Rkpi - Rkk + C * Bd_diff
        
        # We only consider swaps where i < k (to avoid double counting and self-swaps)
        # Also, we should avoid i == k
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)  # upper triangle, i < k
        Delta_masked = np.where(mask, Delta, np.inf)
        
        # Find the best swap
        best_idx = np.unravel_index(np.argmin(Delta_masked), (n, n))
        best_delta = Delta_masked[best_idx]
        
        if best_delta >= -1e-10:
            break  # No improving swap
        
        i, k = best_idx
        u, v = p[i], p[k]
        
        # Perform the swap
        p[i], p[k] = v, u
        
        # Update R
        # R_new[m, :] = R[m, :] + (A[m,i] - A[m,k]) * (B[:, v] - B[:, u])
        # Wait, from earlier derivation:
        # R_new[m, l] = R[m, l] + (a[m,i] - a[m,k]) * (b[l, v] - b[l, u])
        # So R_new[m, :] = R[m, :] + (a[m,i] - a[m,k]) * (b[:, v] - b[:, u])
        
        a_diff = a[:, i] - a[:, k]  # shape (n,)
        b_diff = b[:, v] - b[:, u]  # shape (n,)
        R = R + np.outer(a_diff, b_diff)
        
        current_cost += best_delta
    
    return p, current_cost


def _multi_start_local_search(a, b, total_time=115.0):
    """
    Run local search from multiple starting points, keeping the best result.
    """
    n = a.shape[0]
    start_time = time.time()
    
    best_p = None
    best_cost = np.inf
    
    # Generate multiple initial permutations
    inits = []
    
    # 1. Spectral init
    p_spec = _spectral_init(a, b)
    inits.append(('spectral', p_spec))
    
    # 2. Greedy init
    p_greedy = _greedy_init(a, b)
    inits.append(('greedy', p_greedy))
    
    # 3. Random init
    np.random.seed(42)
    p_rand = np.random.permutation(n).astype(np.int64)
    inits.append(('random', p_rand))
    
    # Run local search on each, with decreasing time budget
    time_per_init = total_time / len(inits)
    
    for name, p_init in inits:
        remaining = total_time - (time.time() - start_time)
        if remaining < 5.0:
            break
        
        t_limit = min(time_per_init, remaining - 2.0)
        if t_limit < 1.0:
            break
        
        p_local, cost = _local_search_2opt(a, b, p_init, time_limit=t_limit)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p_local.copy()
    
    # Use remaining time for additional local search on the best solution
    remaining = total_time - (time.time() - start_time)
    if remaining > 5.0 and best_p is not None:
        p_final, cost_final = _local_search_2opt(a, b, best_p, time_limit=remaining - 3.0)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final
    
    return best_p, best_cost


def solve(a, b):
    """
    Solve the QAP for the given matrices a and b.
    Returns a permutation p of 0..n-1.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before the 150s limit
    
    n = a.shape[0]
    
    # Quick check for small n
    if n <= 1:
        return list(range(n))
    
    # Run multi-start local search
    p, cost = _multi_start_local_search(a, b, total_time=min(140.0, deadline - 5.0))
    
    if p is None:
        # Fallback
        p = _greedy_init(a, b)
    
    # Final check: ensure we're within the deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        pass  # We already stopped due to time limits in _multi_start_local_search
    
    return p.tolist()
# EVOLVE-BLOCK-END
