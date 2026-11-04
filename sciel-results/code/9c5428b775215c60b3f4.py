"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import scipy.optimize as opt

def _compute_cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    n = len(p)
    D = b[np.ix_(p, p)]
    return int(np.sum(a * D))

def _local_search(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search for QAP with efficient delta calculation.
    """
    n = len(p)
    p = p.copy()
    # D[i, j] = b[p[i], p[j]]
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute for delta calculation
    A_sym = a + a.T
    A_diag = np.diag(a)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            # Indices j > i
            js = np.arange(i + 1, n)
            
            # Vectorized delta calculation for all j > i
            # delta(i,j) = sum_k [A_sym[i,k] - A_sym[j,k]] * [D[j,k] - D[i,k]]
            #            + correction for k=i, k=j terms
            #            + (A_diag[i] - A_diag[j]) * (D[j,j] - D[i,i])
            #            + (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
            
            # Compute term1: A_sym[i] @ D.T gives vector of shape (n,)
            term1 = A_sym[i] @ D.T  # shape (n,)
            # term2: scalar
            term2 = A_sym[i] @ D[i]  # scalar
            # term3: for each j in js, sum_k A_sym[j,k] * D[j,k]
            term3 = np.einsum('ij,ij->i', A_sym[js], D[js])  # shape (n-i-1,)
            # term4: A_sym[js] @ D[i] gives vector of shape (n-i-1,)
            term4 = A_sym[js] @ D[i]  # shape (n-i-1,)
            
            # Main sum for each j: term1[j] - term2 - term3[j] + term4[j]
            main_sum = term1[js] - term2 - term3 + term4
            
            # Correction terms:
            # k=i correction for each j in js:
            corr_i = (A_sym[i, i] - A_sym[js, i]) * (D[js, i] - D[i, i])
            
            # k=j correction for each j in js:
            corr_j = (A_sym[i, js] - A_diag[js]) * (D[js, js] - D[i, js])
            
            # Direct swap terms:
            diag_term = (A_diag[i] - A_diag[js]) * (D[js, js] - D[i, i])
            cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = main_sum - corr_i - corr_j + diag_term + cross_term
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost

def _hungarian_reassign(a, b, p):
    """
    Perform a global reassignment using the Hungarian algorithm on a dynamically
    reweighted cost matrix.
    
    The cost of assigning facility i to location j, given current assignment p,
    is approximated by the linearized marginal cost:
    C[i, j] = sum_k a[i, k] * b[j, p[k]] + sum_k a[k, i] * b[p[k], j]
    
    This is equivalent to:
    C[i, j] = (a[i] @ b[j, :].T) + (a[:, i] @ b[:, j].T) ? No.
    
    Let's derive carefully.
    Current cost = sum_{u,v} a[u,v] b[p[u], p[v]]
    
    If we change p[i] to j and p[k] to i (swap i and k in the assignment), 
    the change is complex. 
    
    However, if we treat the assignment as a matching problem where the cost
    of matching facility i to location j is influenced by the other assignments,
    we can define the reduced cost matrix.
    
    A common linearization for QAP (e.g., in tabu search or GRASP) is:
    C[i, j] = sum_k a[i, k] b[j, p[k]] + sum_k a[k, i] b[p[k], j]
    
    Let's verify dimensions.
    a[i, k] is scalar. b[j, p[k]] is scalar. Sum over k.
    This term represents the flow from i to others (k) times the distance from new location j to their current locations.
    Similarly for the second term.
    
    This matrix C can be computed efficiently:
    C = a @ b[np.ix_(range(n), p)] + a.T @ b[np.ix_(p, range(n))]
    
    Wait, b[np.ix_(range(n), p)] is just b[:, p] (columns permuted by p).
    Let B_p = b[:, p]. Then B_p[j, k] = b[j, p[k]].
    Term 1: sum_k a[i, k] * B_p[j, k] = (a @ B_p.T)[i, j] ? 
    (a @ B_p.T)[i, j] = sum_k a[i, k] * B_p.T[k, j] = sum_k a[i, k] * B_p[j, k]. Yes.
    
    Term 2: sum_k a[k, i] * b[p[k], j].
    Let B_p_T = b[p, :]. Then B_p_T[k, j] = b[p[k], j].
    Sum_k a[k, i] * B_p_T[k, j] = (a.T @ B_p_T)[i, j].
    
    So C = (a @ b[:, p].T) + (a.T @ b[p, :])
    
    Then solve linear_sum_assignment(C).
    """
    n = len(a)
    # Efficient matrix multiplication
    # b[:, p] selects columns of b in order p
    B_cols = b[:, p]
    # b[p, :] selects rows of b in order p
    B_rows = b[p, :]
    
    # C[i, j] = sum_k a[i, k] b[j, p[k]] + sum_k a[k, i] b[p[k], j]
    # First part: a @ B_cols.T
    # a is (n, n), B_cols.T is (n, n) -> (n, n)
    part1 = a @ B_cols.T
    # Second part: a.T @ B_rows
    # a.T is (n, n), B_rows is (n, n) -> (n, n)
    part2 = a.T @ B_rows
    
    cost_matrix = part1 + part2
    
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    new_p = np.empty(n, dtype=int)
    new_p[row_ind] = col_ind
    
    return new_p

def _spectral_init(a, b):
    """
    Phase-field-inspired initialization using MDS.
    """
    n = len(a)
    try:
        def mds(dissim, n_dim=2):
            n = len(dissim)
            J = np.eye(n) - np.ones((n, n)) / n
            M = -0.5 * J @ (dissim ** 2) @ J
            eigenvalues, eigenvectors = np.linalg.eigh(M)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            V = eigenvectors[:, :n_dim]
            D = np.sqrt(np.maximum(eigenvalues[:n_dim], 0))
            coords = V * D
            return coords
        
        loc_coords = mds(b, n_dim=2)
        fac_coords = mds(-a, n_dim=2)
        
        cost_matrix = np.sum((fac_coords[:, np.newaxis, :] - loc_coords[np.newaxis, :, :]) ** 2, axis=2)
        
        row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=int)
        p[row_ind] = col_ind
        
        return p
        
    except Exception:
        return np.random.permutation(n)

def _hungarian_init(a, b):
    """
    Standard Hungarian initialization on surrogate cost.
    """
    n = len(a)
    cost_matrix = a @ b + a.T @ b.T
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=int)
    p[row_ind] = col_ind
    return p

def _perturb(p, k):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer for safety
    
    best_p = None
    best_cost = np.inf
    
    # Helper to update best
    def update_best(p, cost):
        nonlocal best_p, best_cost
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()

    # Phase 1: Multiple initializations
    # 1. Hungarian Init
    if time.time() - start_time < time_budget * 0.2:
        p_init = _hungarian_init(a, b)
        p_ls, cost_ls = _local_search(a, b, p_init, time_budget * 0.1, start_time)
        update_best(p_ls, cost_ls)
        
        # Apply Hungarian Reassignment to the local optimum
        if time.time() - start_time < time_budget * 0.25:
            p_hung = _hungarian_reassign(a, b, p_ls)
            p_ls2, cost_ls2 = _local_search(a, b, p_hung, time_budget * 0.05, start_time)
            update_best(p_ls2, cost_ls2)

    # 2. Spectral Init
    if time.time() - start_time < time_budget * 0.35:
        p_init = _spectral_init(a, b)
        p_ls, cost_ls = _local_search(a, b, p_init, time_budget * 0.1, start_time)
        update_best(p_ls, cost_ls)
        
        if time.time() - start_time < time_budget * 0.4:
            p_hung = _hungarian_reassign(a, b, p_ls)
            p_ls2, cost_ls2 = _local_search(a, b, p_hung, time_budget * 0.05, start_time)
            update_best(p_ls2, cost_ls2)

    # 3. Random Inits
    for _ in range(2):
        if time.time() - start_time > time_budget * 0.5:
            break
        p_init = np.random.permutation(n)
        p_ls, cost_ls = _local_search(a, b, p_init, time_budget * 0.1, start_time)
        update_best(p_ls, cost_ls)
        
        if time.time() - start_time < time_budget * 0.55:
            p_hung = _hungarian_reassign(a, b, p_ls)
            p_ls2, cost_ls2 = _local_search(a, b, p_hung, time_budget * 0.05, start_time)
            update_best(p_ls2, cost_ls2)

    # Phase 2: ILS with Hungarian Reassignment
    # The key idea: Local Search -> Hungarian Reassignment -> Local Search
    # This allows jumping to a different region of the search space.
    
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 10:
        return best_p.tolist()
        
    consecutive_no_improve = 0
    base_perturb = 2
    max_perturb = 20
    
    while time.time() - start_time < time_budget - 5:
        # Check time
        if time.time() - start_time > time_budget - 5:
            break
            
        # Adaptive perturbation
        if consecutive_no_improve > 10:
            perturb_k = min(max_perturb, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
            
        p_pert = _perturb(best_p, perturb_k)
        
        # Step 1: Local Search
        time_for_ls1 = min(5.0, (time_budget - 5 - (time.time() - start_time)) * 0.4)
        p_ls1, cost_ls1 = _local_search(a, b, p_pert, time_for_ls1, start_time)
        
        # Step 2: Hungarian Reassignment
        # This is fast, so we can always do it if time permits
        if time.time() - start_time < time_budget - 5:
            p_hung = _hungarian_reassign(a, b, p_ls1)
            
            # Step 3: Local Search on Hungarian result
            time_for_ls2 = min(5.0, (time_budget - 5 - (time.time() - start_time)) * 0.6)
            p_ls2, cost_ls2 = _local_search(a, b, p_hung, time_for_ls2, start_time)
            
            # Check improvement
            if cost_ls2 < best_cost:
                update_best(p_ls2, cost_ls2)
                consecutive_no_improve = 0
                base_perturb = max(2, base_perturb - 1)
            elif cost_ls1 < best_cost:
                # In case Hungarian made it worse, but LS1 was better than global best
                update_best(p_ls1, cost_ls1)
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1
                # Increase perturbation on stagnation
                if consecutive_no_improve % 5 == 0:
                    base_perturb = min(max_perturb, base_perturb + 2)

    # Final Polish
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time > 2:
        # One last local search on the best solution found
        p_final, cost_final = _local_search(a, b, best_p, remaining_time - 1, start_time)
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final

    return best_p.tolist()
# EVOLVE-BLOCK-END
