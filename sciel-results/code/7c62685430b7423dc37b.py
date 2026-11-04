"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import scipy.optimize as opt
from math import log, exp

def _qap_cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    n = len(p)
    D = b[np.ix_(p, p)]
    return int(np.sum(a * D))

def _local_search_2opt(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search for QAP with vectorized delta computation.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    A_sym = a + a.T
    A_diag = np.diag(a).copy()
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            js = np.arange(i + 1, n)
            
            diff_A = A_sym[i] - A_sym[js]
            diff_D = D[js] - D[i]
            
            S = np.sum(diff_A * diff_D, axis=1)
            
            coeff_i = A_sym[i, i] - A_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            coeff_j = A_sym[i, js] - A_diag[js]
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            diag_term = (A_diag[i] - A_diag[js]) * (D[js, js] - D[i, i])
            
            cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = S - term_i - term_j + diag_term + cross_term
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                p[i], p[j] = p[j], p[i]
                
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost

def _local_search_or_opt(a, b, p, time_limit, start_time):
    """
    Or-opt local search: move a single facility to a different position.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            # Try moving facility at position i to position j
            # The delta for moving i to j (j != i) is complex, so we use a simplified
            # version: swap i with each j and check if it's better (which is what 2-opt does).
            # For true or-opt (move, not swap), we need a different delta formula.
            # Let's stick to 2-opt for reliability, but add a "block move" variant.
            
            # Actually, let's implement a simple "move" by shifting.
            # But for now, let's rely on 2-opt which is solid.
            # We can add 3-opt later if needed.
            pass
        
        # If 2-opt found no improvement, try a random restart or more aggressive search?
        # For now, break if no improvement in a full pass.
        # Since the loop above doesn't track improvement for or-opt specifically,
        # let's just run 2-opt again with a different randomization?
        # No, let's keep it simple: just run 2-opt.
        break  # Placeholder: this function currently just delegates to 2-opt logic or exits
    
    return p, current_cost

def _hungarian_init(a, b):
    """
    Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix.
    """
    n = len(a)
    cost_matrix = a @ b + a.T @ b.T
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=int)
    p[row_ind] = col_ind
    return p

def _eigen_init(a, b):
    """
    Generate an initial assignment using eigenvector alignment.
    This is based on the idea that the QAP can be approximated by aligning the
    principal components of the flow and distance matrices.
    """
    n = len(a)
    # Center the matrices
    a_centered = a - np.mean(a, axis=1, keepdims=True) - np.mean(a, axis=0, keepdims=True) + np.mean(a)
    b_centered = b - np.mean(b, axis=1, keepdims=True) - np.mean(b, axis=0, keepdims=True) + np.mean(b)
    
    # Get top eigenvectors
    try:
        _, vecs_a = np.linalg.eigh(a_centered)
        _, vecs_b = np.linalg.eigh(b_centered)
        
        # Use the top k eigenvectors (e.g., 50% of n)
        k = n // 2
        # Align the eigenvectors: the sign is ambiguous, so we can flip them
        # to maximize the inner product.
        U = vecs_a[:, :k]
        V = vecs_b[:, :k]
        
        # The assignment is based on the similarity between the rows of U and V?
        # Actually, this is tricky. A common heuristic is to use the first eigenvector
        # to order the facilities and locations, then use a nearest-neighbor matching.
        
        # Let's try a simpler approach: sort facilities by their projection on the first eigenvector
        # and locations by their projection on the first eigenvector, then match them in order.
        
        eig_a = vecs_a[:, 0]
        eig_b = vecs_b[:, 0]
        
        # Sort facilities by their score
        order_a = np.argsort(eig_a)
        order_b = np.argsort(eig_b)
        
        # Match them
        p = np.empty(n, dtype=int)
        for i in range(n):
            p[order_a[i]] = order_b[i]
            
        return p
    except Exception:
        # Fallback to random
        return np.random.permutation(n)

def _phase_field_init(a, b, time_limit, start_time):
    """
    Attempt a continuous relaxation approach.
    Define a matrix X where X[i, j] is the 'density' of facility i at location j.
    Initially, X is uniform. We want to minimize a surrogate energy.
    This is computationally expensive, so we use it only if time permits and as a diverse initializer.
    """
    n = len(a)
    # This is a placeholder for a more complex continuous method.
    # Given the time constraints, we might skip this or use a very simplified version.
    # For now, let's return a random permutation to keep the code path distinct but safe.
    return np.random.permutation(n)

def _perturb(p, k):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def _reoptimize_hungarian_segment(a, b, p, start_idx, end_idx):
    """
    Re-optimize a segment of the permutation using the Hungarian algorithm
    on the induced sub-problem. This can help escape local optima.
    """
    n = len(p)
    if end_idx - start_idx < 2:
        return p, _qap_cost(a, b, p)
    
    # Extract sub-matrices
    facilities = np.arange(start_idx, end_idx)
    # The locations currently occupied by these facilities
    current_locs = p[facilities]
    
    # The cost of assigning these facilities to these locations, considering interactions
    # with all other facilities (fixed) and within the segment.
    
    # This is complex because the cost of placing facility i at location j depends on
    # where all other facilities are.
    # Let's approximate by only considering interactions within the segment + the fixed cost.
    
    # Fixed cost: interactions of these facilities with all facilities outside the segment.
    # This is constant regardless of the assignment within the segment? No, it depends on the locations.
    # If facility i is at location j, its cost with facility k (outside) at location p[k] is a[i,k]*b[j,p[k]].
    # So the fixed cost for facility i depends on j.
    
    # Let's build the cost matrix for the sub-problem.
    size = end_idx - start_idx
    cost_matrix = np.zeros((size, size))
    
    # For each facility i in the segment and each location j in the segment (from current_locs)
    for idx_i, i in enumerate(facilities):
        for idx_j, j in enumerate(current_locs):
            # Cost contribution from interactions with facilities inside the segment
            # This is hard to compute without knowing the final assignment of other facilities in the segment.
            # Let's use a surrogate: (a[i, i_in_seg] * b[j, j_in_seg]).sum()
            # This is what the Hungarian init does globally.
            
            # Let's use the global surrogate cost restricted to this subset.
            # cost[i, j] = sum_k a[i, k] * b[j, k] + sum_k a[k, i] * b[k, j]
            # But we only care about k in the segment? No, the interactions with outside are also important.
            
            # Let's compute the full surrogate cost for this facility-location pair.
            # cost[i, j] = np.sum(a[i, :] * b[j, :]) + np.sum(a[:, i] * b[:, j])
            # This is expensive: O(n) per pair, O(n^3) total for the sub-problem.
            # With n=256, and segment size ~50, this is 50*50*256 = 640k operations. Acceptable.
            
            c1 = np.sum(a[i, :] * b[j, :])
            c2 = np.sum(a[:, i] * b[:, j])
            cost_matrix[idx_i, idx_j] = c1 + c2
    
    # Solve the assignment problem for this sub-problem
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    
    # Update the permutation
    new_locs = current_locs[col_ind]
    for idx_i, i in enumerate(facilities):
        p[i] = new_locs[row_ind[idx_i]]
        
    return p, _qap_cost(a, b, p)

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    # 1. Hungarian Initialization
    p_hung = _hungarian_init(a, b)
    ls_limit = time_budget * 0.15
    p_ls, cost_ls = _local_search_2opt(a, b, p_hung, ls_limit, start_time)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # 2. Eigen Initialization
    if time.time() - start_time < time_budget * 0.3:
        p_eig = _eigen_init(a, b)
        ls_limit = time_budget * 0.1
        p_ls_eig, cost_ls_eig = _local_search_2opt(a, b, p_eig, ls_limit, start_time)
        if cost_ls_eig < best_cost:
            best_cost = cost_ls_eig
            best_p = p_ls_eig.copy()

    # 3. ILS phase with adaptive perturbation and re-optimization
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 5:
        return best_p.tolist()
        
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() - start_time < time_budget - 2:
        if time.time() - start_time > time_budget - 2:
            break
            
        # Adaptive perturbation
        if consecutive_no_improve > 10:
            perturb_k = min(20, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
            
        p_pert = _perturb(best_p, perturb_k)
        
        # Sometimes, instead of just perturbing, re-optimize a random segment
        if random.random() < 0.1 and n > 10:
            seg_size = random.randint(10, min(50, n // 2))
            start_idx = random.randint(0, n - seg_size)
            end_idx = start_idx + seg_size
            p_pert, _ = _reoptimize_hungarian_segment(a, b, p_pert, start_idx, end_idx)
            
        duration = (time_budget - 2) - (time.time() - start_time)
        if duration < 1:
             break
             
        p_ls, cost_ls = _local_search_2opt(a, b, p_pert, duration, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
