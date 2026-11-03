"""QAP solver for tho150 using GRASP + first-improvement 2-opt."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Convert to float64 for faster arithmetic
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    
    def compute_cost(p):
        """Compute exact cost for permutation p."""
        return np.sum(a_f * b_f[np.ix_(p, p)])
    
    def get_delta(p, i, j):
        """
        Compute delta for swapping p[i] and p[j].
        Uses vectorized O(n) operations.
        """
        li, lj = p[i], p[j]
        
        # diff_a_row[k] = a[i,k] - a[j,k]
        # diff_b_col[k] = b[lj, p[k]] - b[li, p[k]]
        # diff_a_col[k] = a[k,i] - a[k,j]  
        # diff_b_row[k] = b[p[k], lj] - b[p[k], li]
        
        diff_a_row = a_f[i] - a_f[j]
        diff_b_col = b_f[lj, p] - b_f[li, p]
        diff_a_col = a_f[:, i] - a_f[:, j]
        diff_b_row = b_f[p, lj] - b_f[p, li]
        
        delta = np.dot(diff_a_row, diff_b_col) + np.dot(diff_a_col, diff_b_row)
        
        # Correction for k=i and k=j (the self-interaction terms)
        # The dot products above include k=i and k=j incorrectly
        # We need to subtract those and add the correct corner terms
        
        # For k=i: the formula gives terms that don't match the actual swap effect
        # For k=j: same issue
        
        # Subtract the incorrect contributions
        # delta_vec[i] = diff_a_row[i]*diff_b_col[i] + diff_a_col[i]*diff_b_row[i]
        # delta_vec[j] = diff_a_row[j]*diff_b_col[j] + diff_a_col[j]*diff_b_row[j]
        
        delta -= (diff_a_row[i] * diff_b_col[i] + diff_a_col[i] * diff_b_row[i])
        delta -= (diff_a_row[j] * diff_b_col[j] + diff_a_col[j] * diff_b_row[j])
        
        # Add correct corner terms:
        # Old: a[i,i]*b[li,li] + a[i,j]*b[li,lj] + a[j,i]*b[lj,li] + a[j,j]*b[lj,lj]
        # New: a[i,i]*b[lj,lj] + a[i,j]*b[lj,li] + a[j,i]*b[li,lj] + a[j,j]*b[li,li]
        
        delta += a_f[i, i] * (b_f[lj, lj] - b_f[li, li])
        delta += a_f[i, j] * (b_f[lj, li] - b_f[li, lj])
        delta += a_f[j, i] * (b_f[li, lj] - b_f[lj, li])
        delta += a_f[j, j] * (b_f[li, li] - b_f[lj, lj])
        
        return delta
    
    def first_improvement_2opt(p, current_cost, time_limit):
        """
        First-improvement 2-opt local search.
        Scans pairs in order, applies first improving swap found.
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n):
                if time.time() > time_limit:
                    return p, current_cost
                pi = p[i]
                for j in range(i + 1, n):
                    delta = get_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
                        break
                if improved:
                    break
        
        return p, current_cost
    
    def rank_matching_initial():
        """
        Initial solution: rank heavy facilities to central locations.
        """
        a_row_sums = a_f.sum(axis=1)
        b_row_sums = b_f.sum(axis=1)
        
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_row_sums, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p
    
    def grasp_construction(randomness=0.5):
        """
        GRASP-style construction: randomized greedy with restricted candidate list.
        At each step, pick from top-K candidates with some randomness.
        """
        p = np.full(n, -1, dtype=np.int64)
        assigned_facs = np.zeros(n, dtype=bool)
        assigned_locs = np.zeros(n, dtype=bool)
        
        # marginal[i][j] = incremental cost of assigning facility i to location j
        # given current partial assignment
        marginal = np.zeros((n, n), dtype=np.float64)
        
        for step in range(n):
            # Mask assigned facilities and locations
            mask = np.zeros((n, n), dtype=bool)
            mask[:, assigned_locs] = True
            mask[assigned_facs, :] = True
            
            # Get candidate costs
            costs = marginal.copy()
            costs[mask] = np.inf
            
            # Find top-K candidates (K depends on randomness)
            # Lower randomness = smaller K = more greedy
            K = max(5, int(n * randomness))
            
            # Get indices of K smallest costs
            flat_costs = costs.ravel()
            # Use argpartition for efficiency
            k_smallest_idx = np.argpartition(flat_costs, K)[:K]
            
            # Get the actual costs of these candidates
            candidate_costs = flat_costs[k_smallest_idx]
            
            # Weighted random selection: prefer lower costs
            # Use softmax-like weights: weight = 1 / (cost + eps)
            eps = 1e-10
            weights = 1.0 / (candidate_costs + eps)
            weights /= weights.sum()
            
            # Select one candidate
            selected_idx = np.random.choice(len(candidate_costs), p=weights)
            selected_flat_idx = k_smallest_idx[selected_idx]
            
            i, j = np.unravel_index(selected_flat_idx, (n, n))
            
            # Assign facility i to location j
            p[i] = j
            assigned_facs[i] = True
            assigned_locs[j] = True
            
            # Update marginal costs
            # When assigning f=i to l=j:
            # For all unassigned (u, v):
            #   marginal[u,v] += a[u,i]*b[v,j] + a[i,u]*b[j,v]
            # (The a[i,i]*b[j,j] term is added to cost directly, not marginal)
            
            marginal += np.outer(a_f[:, i], b_f[j, :]) + np.outer(a_f[i, :], b_f[:, j])
        
        return p
    
    # --- Main Algorithm ---
    
    # Phase 1: Get a good initial solution via rank matching
    p_init = rank_matching_initial()
    cost_init = compute_cost(p_init)
    
    best_p = p_init.copy()
    best_cost = cost_init
    
    # Phase 2: Local search on initial solution
    time_left = deadline - time.time() - 5.0
    if time_left > 1.0:
        ls_limit = time.time() + min(30.0, time_left * 0.4)
        p_ls, cost_ls = first_improvement_2opt(p_init.copy(), cost_init, ls_limit)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Phase 3: GRASP restarts
    # Each restart: GRASP construction + local search
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 5.0:
        # Vary randomness for diversity
        randomness = rng.uniform(0.3, 0.8)
        
        # GRASP construction
        p_grasp = grasp_construction(randomness=randomness)
        cost_grasp = compute_cost(p_grasp)
        
        # Check if this is already better
        if cost_grasp < best_cost:
            best_cost = cost_grasp
            best_p = p_grasp.copy()
        
        # Local search on the GRASP solution
        time_left = deadline - time.time() - 5.0
        if time_left < 1.0:
            break
        
        # Limit local search time per restart
        ls_time = min(20.0, time_left * 0.7)
        ls_limit = time.time() + ls_time
        
        p_ls, cost_ls = first_improvement_2opt(p_grasp.copy(), cost_grasp, ls_limit)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
