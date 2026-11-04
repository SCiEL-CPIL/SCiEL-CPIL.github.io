"""
Spectral Quadratic Linearization (QL) + Multi-Restart Simulated Annealing
Strategy:
1. Decompose flow matrix A into rank-1 terms using its top eigenvectors.
2. Solve a sequence of linear assignment problems (Hungarian) to minimize the surrogate cost.
3. Use the best surrogate solution as the starting point for a vectorized Simulated Annealing.
4. Perform multiple restarts if time permits to escape local optima.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    O(n) vectorized computation.
    """
    pi = p[i]
    pj = p[j]
    
    # Direct terms (i,j pair)
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Diagonal terms
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Cross terms with all other k
    # Term 1: sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract k=i and k=j contributions
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    # Term 2: sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract k=i and k=j contributions
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _spectral_ql_init(a, b, num_components=10):
    """
    Quadratic Linearization via Spectral Decomposition.
    1. Compute top eigenvectors of A (symmetrized if necessary for stability, though A is not necessarily symmetric).
       For QAP, A is the flow matrix. We use the symmetric part (A + A^T)/2 for eigen-decomposition 
       to get meaningful spatial modes, or just use A directly if we treat it as linear operator.
       Standard QL uses the decomposition of the quadratic form.
       Cost = sum a_ij b_pi_pj.
       We approximate A by sum_{k=1}^K lambda_k v_k v_k^T (for symmetric A).
       If A is not symmetric, we can use SVD or just top eigenvectors of (A+A^T)/2.
       Let's use the symmetric part for stability in eigen-decomp, as flow is often effectively symmetric 
       or the antisymmetric part contributes less to the global structure in many instances.
       Actually, for wil100, A is not symmetric. But the dominant structure is often captured by the symmetric part.
       Let's try decomposing A itself using SVD or just taking top eigenvectors of (A+A^T)/2.
    """
    n = len(a)
    
    # Symmetrize A for spectral decomposition stability
    A_sym = (a + a.T) / 2.0
    
    # Compute top eigenvectors
    # For n=100, full eigendecomposition is fast enough
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    
    # Sort by magnitude of eigenvalues (descending)
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    best_p = None
    best_cost = np.inf
    
    # Try a few different numbers of components
    for k in range(1, min(num_components + 1, n)):
        # Reconstruct surrogate cost matrix C
        # C[i, j] = sum_{m=1}^k lambda_m * (v_m[i] * v_m[j]) * b[i, j] ? 
        # No, the surrogate assignment cost for facility i at location j should be:
        # Cost[i, j] = sum_{m=1}^k lambda_m * v_m[i] * (sum_l v_m[l] * b[j, l]) ?
        # Let's derive:
        # Approx A = V_k Lambda V_k^T
        # Cost = sum_{i,j} (sum_m lambda_m v_mi v_mj) b_pi_pj
        # = sum_m lambda_m (sum_i v_mi x_i) (sum_j v_mj x_j) where x is assignment?
        # This is tricky to map directly to linear assignment C[i,j] such that sum C[i, p_i] approximates cost.
        
        # Standard QL approach:
        # Minimize sum_{i,j} a_ij b_pi_pj
        # Approximate a_ij by sum_{m=1}^k u_im u_jm
        # Then cost approx sum_m (sum_i u_im b_pi_pj ... )
        
        # Alternative simpler QL:
        # Use the row sums of the reconstructed A_k to form the assignment cost.
        # Let A_k = V_k Lambda_k V_k^T
        # The "effective" flow from facility i to all others is roughly row sum of A_k.
        # But we need to match with B.
        
        # Let's use the standard method:
        # C[i, j] = sum_{m=1}^k lambda_m * v_m[i] * (V_k[:, m] @ b[j, :])
        # This assumes the "location j" interacts with the "mode m" via the projection of row j of B onto v_m.
        
        # Compute projections of B rows onto eigenvectors
        # proj_b[m, j] = dot(v_m, b[j, :])
        V_k = eigvecs[:, :k]
        Lambda_k = eigvals[:k]
        
        # proj_b shape: (k, n)
        proj_b = V_k.T @ b  # (k, n)
        
        # C[i, j] = sum_{m=1}^k Lambda_k[m] * V_k[i, m] * proj_b[m, j]
        # C = V_k @ diag(Lambda_k) @ proj_b
        C = (V_k * Lambda_k) @ proj_b
        
        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        cost = _qap_cost(a, b, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            
    return best_p, best_cost

def _sa_polish(a, b, p, time_limit):
    """Simulated annealing polish phase with vectorized delta."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    # Cooling schedule
    t0 = 5000.0
    t_final = 0.01
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        frac = min(elapsed / time_limit, 1.0)
        temp = t0 * (t_final / t0) ** frac
        
        # Process a batch of swaps to amortize time check
        # Adjust batch size based on remaining time? Keep fixed for simplicity.
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
                
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = _swap_delta(a, b, p, i, j)
            
            # Acceptance
            if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
    
    return best_p, best_cost

def _local_search(a, b, p, time_limit):
    """First-improvement local search."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Randomize order of checks to avoid bias? 
        # For n=100, 4950 pairs. Checking all is ~0.5s.
        # We do a single pass of first improvement.
        
        # Shuffle indices to ensure we don't always start from (0,1)
        indices = np.random.permutation(n)
        for i_idx in range(n):
            for j_idx in range(i_idx + 1, n):
                i = indices[i_idx]
                j = indices[j_idx]
                
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
            
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    deadline = 115.0  # Safety margin before 120s limit
    start = time.time()
    
    np.random.seed(42)
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Spectral QL Initialization
    # This is much faster than GNN and theoretically grounded.
    p_init, cost_init = _spectral_ql_init(a, b, num_components=15)
    best_p = p_init.copy()
    best_cost = cost_init
    
    # Phase 2: Multi-Restart SA + LS
    # We have ~110s left.
    # Let's do 5 restarts.
    # Each restart: 
    #   - Perturb best solution (random swaps)
    #   - SA for short time
    #   - LS for short time
    #   - Keep best
    
    num_restarts = 5
    time_per_restart = (deadline - (time.time() - start) - 5.0) / num_restarts
    time_per_restart = max(5.0, min(20.0, time_per_restart))
    
    for r in range(num_restarts):
        if time.time() - start > deadline - 10.0:
            break
            
        # Perturb current best
        p_current = best_p.copy()
        # Perform 10-20 random swaps
        num_swaps = 10 + np.random.randint(10)
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p_current[i], p_current[j] = p_current[j], p_current[i]
        
        # SA Polish
        sa_time = time_per_restart * 0.7
        p_sa, cost_sa = _sa_polish(a, b, p_current, sa_time)
        
        # LS Polish
        ls_time = time_per_restart * 0.3
        p_ls, cost_ls = _local_search(a, b, p_sa, ls_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            
    # Phase 3: Final Deep Polish on the very best solution found
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 5.0:
        p_final, cost_final = _local_search(a, b, best_p, time_remaining - 2.0)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()
            
    return best_p.tolist()

# EVOLVE-BLOCK-END
