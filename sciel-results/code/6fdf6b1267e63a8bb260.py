"""
Combined Iterated Local Search + Simulated Annealing solver for QAPLIB wil100.
Strategy:
1. Fast O(n) incremental delta evaluation for 2-opt swaps using precomputed vectors.
2. Multi-start: spectral initialization + rank-matching + random restarts.
3. Iterated Local Search:
   - First-improvement local search until local optimum.
   - Perturbation (k random swaps) to escape local optimum.
   - Accept if perturbed solution improves best, or with SA probability.
4. Final polish with SA from the best solution found.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

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
    
    # Diagonal terms (usually 0 for QAP but handle generally)
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Cross terms with all other k
    # Term 1: sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    # Term 2: sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    
    # Compute over all k, then subtract k=i and k=j contributions
    
    # Term 1 over all k:
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract k=i: (a[i,i]-a[j,i]) * (b[pj,pi]-b[pi,pi])
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    # Subtract k=j: (a[i,j]-a[j,j]) * (b[pj,pj]-b[pi,pj])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    # Term 2 over all k:
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract k=i: (a[i,i]-a[i,j]) * (b[pi,pj]-b[pi,pi])
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    # Subtract k=j: (a[j,i]-a[j,j]) * (b[pj,pj]-b[pj,pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _spectral_initial(a, b, n_components=20):
    """
    Create an initial permutation using spectral decomposition.
    Aligns eigenvectors of the flow matrix with eigenvectors of the distance matrix
    to place high-flow facilities in high-distance locations.
    """
    n = len(a)
    
    # Symmetrize for eigenvalue decomposition
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # Get top eigenvectors
    try:
        eigenvalues_a, eigenvectors_a = np.linalg.eigh(a_sym)
        eigenvalues_b, eigenvectors_b = np.linalg.eigh(b_sym)
        
        # Sort by eigenvalue (largest first)
        idx_a = np.argsort(eigenvalues_a)[::-1]
        idx_b = np.argsort(eigenvalues_b)[::-1]
        
        eigenvectors_a = eigenvectors_a[:, idx_a]
        eigenvectors_b = eigenvectors_b[:, idx_b]
        
        # Use top k components
        k = min(n_components, n)
        V_a = eigenvectors_a[:, :k]
        V_b = eigenvectors_b[:, :k]
        
        # Project facilities and locations onto the shared eigenspace
        # For each facility i, compute its "position" in the spectral space
        # and for each location j, compute its "position"
        
        # Use the first few eigenvectors to create a low-dimensional embedding
        # Facility i's embedding: [V_a[i,0], V_a[i,1], ..., V_a[i,k-1]]
        # Location j's embedding: [V_b[j,0], V_b[j,1], ..., V_b[j,k-1]]
        
        # We want to assign facility i to location p[i] such that
        # high-flow facilities (large projection on high eigenvectors of a)
        # go to high-distance locations (large projection on high eigenvectors of b)
        
        # Simple approach: match by rank of projection onto the top eigenvector
        # But a better approach is to use the full spectral embedding
        
        # Compute projection scores
        # For facility i: score_a[i] = sum of |V_a[i,m]| * eigenvalues_a[m] for top m
        # For location j: score_b[j] = sum of |V_b[j,m]| * eigenvalues_b[m] for top m
        
        scores_a = np.zeros(n)
        scores_b = np.zeros(n)
        for m in range(k):
            scores_a += np.abs(V_a[:, m]) * abs(eigenvalues_a[m])
            scores_b += np.abs(V_b[:, m]) * abs(eigenvalues_b[m])
        
        # Alternative: use the sign pattern from the top eigenvector
        # Facilities with positive projection on top eigenvector of a
        # should go to locations with positive projection on top eigenvector of b
        
        # Use Hungarian algorithm or greedy matching based on spectral alignment
        # For simplicity, use rank matching on the combined spectral score
        
        # Actually, let's use a more direct approach:
        # Create a cost matrix in spectral space and solve assignment
        
        # Embed facilities in R^k using top eigenvectors of a
        # Embed locations in R^k using top eigenvectors of b
        # Then assign each facility to the closest location
        
        # But this is O(n^2 * k) which is fine for n=100
        
        # Compute pairwise distances in spectral space
        # dist[i,j] = ||V_a[i,:] - V_b[j,:]||^2
        # But we want to MINIMIZE the QAP cost, not spectral distance
        
        # Better approach: use the spectral structure to create a good initial assignment
        # by aligning the principal components
        
        # For the top eigenvector of a (call it u_a) and top eigenvector of b (call it u_b):
        # Facilities with u_a[i] > 0 should go to locations with u_b[j] > 0
        # Facilities with u_a[i] < 0 should go to locations with u_b[j] < 0
        
        # More generally, use multiple eigenvectors to create a finer partition
        
        # Let's use a simple spectral matching:
        # Sort facilities by their projection on the top eigenvector of a
        # Sort locations by their projection on the top eigenvector of b
        # Match in order
        
        # Use the top eigenvector
        top_facility_scores = V_a[:, 0]
        top_location_scores = V_b[:, 0]
        
        # Sort facilities by their score (high to low)
        facility_order = np.argsort(-top_facility_scores, kind="stable")
        # Sort locations by their score (high to low)
        location_order = np.argsort(-top_location_scores, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        p[facility_order] = location_order
        
        # Refine using additional eigenvectors
        # For the second eigenvector, within each half, match by second component
        # This is a recursive refinement
        
        # Simpler: use the full spectral embedding and do nearest-neighbor assignment
        # Compute the assignment that minimizes sum of squared distances in spectral space
        
        # Actually, let's just use the rank matching on the top eigenvector
        # and let local search refine it
        
    except np.linalg.LinAlgError:
        # Fallback to rank matching
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
    
    return p

def _greedy_initial(a, b):
    """Create an initial permutation using rank-matching heuristic."""
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _local_search(a, b, p, time_limit, start_time):
    """
    First-improvement local search: repeatedly pick a random swap,
    accept if it improves. Continue until time runs out or no improvement
    found after many tries.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _swap_delta(a, b, p, i, j)
        
        if delta < 0:
            p[i], p[j] = p[j], p[i]
            current_cost += int(delta)
            no_improve = 0
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
    
    return best_p, best_cost

def _perturb(p, k):
    """Apply k random swaps to p, returning a copy."""
    n = len(p)
    p_new = p.copy()
    for _ in range(k):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i != j:
            p_new[i], p_new[j] = p_new[j], p_new[i]
    return p_new

def _sa_polish(a, b, p, time_limit, start_time):
    """
    Simulated Annealing polish from a good starting point.
    Uses time-based cooling schedule.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    T_start = 500.0
    T_end = 1.0
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        # Linear cooling based on fraction of time used
        # We don't know total iterations, so use a fixed schedule based on time
        # T decreases from T_start to T_end over the time_limit
        frac = min(1.0, elapsed / max(time_limit, 1.0))
        T = T_start * (T_end / T_start) ** frac
        
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _swap_delta(a, b, p, i, j)
        
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
            p[i], p[j] = p[j], p[i]
            current_cost += int(delta)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0  # Safety margin before 120s limit
    start_time = time.time()
    
    np.random.seed(42)  # Fixed seed for reproducibility within a run
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Multi-start with local search
    # Start with spectral initialization
    p = _spectral_initial(a, b, n_components=10)
    p, cost = _local_search(a, b, p, 5.0, start_time)
    best_p = p.copy()
    best_cost = cost
    
    # Also try rank-matching initialization
    p_rank = _greedy_initial(a, b)
    p_rank, cost_rank = _local_search(a, b, p_rank, 3.0, start_time)
    if cost_rank < best_cost:
        best_cost = cost_rank
        best_p = p_rank.copy()
    
    # Try several random restarts with local search
    num_restarts = 15
    for r in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > total_deadline - 30:
            break
        
        p_rand = np.random.permutation(n).astype(np.int64)
        # Short local search from random start
        ls_time = min(3.0, total_deadline - elapsed - 25)
        if ls_time < 0.5:
            break
        
        p_ls, cost_ls = _local_search(a, b, p_rand.copy(), ls_time, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Phase 2: Iterated Local Search from best solution
    elapsed = time.time() - start_time
    if elapsed < total_deadline - 20:
        p_ils = best_p.copy()
        cost_ils = best_cost
        perturbation_k = 3
        
        while time.time() - start_time < total_deadline - 15:
            # Perturb
            p_pert = _perturb(p_ils, perturbation_k)
            
            # Local search from perturbed
            ls_time = min(5.0, total_deadline - (time.time() - start_time) - 10)
            if ls_time < 0.5:
                break
            
            p_ls, cost_ls = _local_search(a, b, p_pert, ls_time, start_time)
            
            # Acceptance: if better than current ILS solution, or with SA probability
            if cost_ls < cost_ils:
                p_ils = p_ls
                cost_ils = cost_ls
                # Reduce perturbation on improvement
                perturbation_k = max(2, perturbation_k - 1)
            else:
                # SA acceptance: accept worse solution with some probability
                # to escape local minima
                delta = cost_ls - cost_ils
                T = 100.0  # Fixed temperature for ILS acceptance
                if np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                    p_ils = p_ls
                    cost_ils = cost_ls
                # Increase perturbation on stagnation
                perturbation_k = min(10, perturbation_k + 1)
            
            if cost_ils < best_cost:
                best_cost = cost_ils
                best_p = p_ils.copy()
    
    # Phase 3: Final SA polish from best solution
    elapsed = time.time() - start_time
    if elapsed < total_deadline - 3:
        sa_time = total_deadline - elapsed - 2
        p_final, cost_final = _sa_polish(a, b, best_p.copy(), sa_time, start_time)
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
