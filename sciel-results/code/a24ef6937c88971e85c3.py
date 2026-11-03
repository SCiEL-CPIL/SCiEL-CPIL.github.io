"""QAP solver for tho150 combining vectorized local search with diverse GRASP restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    """Solve QAP: assign facilities to locations to minimize sum a[i,j]*b[p[i],p[j]].
    
    Args:
        a: flow matrix (n x n) int64
        b: distance matrix (n x n) int64
    Returns:
        p: permutation list where facility i goes to location p[i]
    """
    n = len(a)
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Convert to float64 for faster arithmetic in delta calculations
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    
    # Precompute row sums for initialization heuristics
    a_row_sums = a_f.sum(axis=1)
    b_row_sums = b_f.sum(axis=1)
    
    def compute_cost(p):
        """Compute exact cost of permutation p."""
        M = b_f[np.ix_(p, p)]
        return np.sum(a_f * M)
    
    def vectorized_local_search(p, deadline):
        """Perform first-improvement 2-swap local search using vectorized delta calculation.
        
        Returns the improved permutation and its cost.
        """
        p = p.copy()
        cost = compute_cost(p)
        
        # Precompute a and b rows/cols for faster access if needed, 
        # but we'll index directly in the loop for memory efficiency.
        
        while time.time() < deadline - 2.0:
            improved = False
            
            for i in range(n):
                if time.time() > deadline - 2.0:
                    break
                
                pi = p[i]
                # Precompute vectors involving location pi and current permutation p
                # B_pi_P[k] = b[pi, p[k]]
                B_pi_P = b_f[pi, p]
                # B_P_pi[k] = b[p[k], pi]
                B_P_pi = b_f[p, pi]
                
                # a[i, :] and a[:, i]
                a_i_row = a_f[i, :]
                a_i_col = a_f[:, i]
                
                for j in range(i + 1, n):
                    pj = p[j]
                    if pi == pj:
                        continue
                    
                    # Precompute vectors involving location pj
                    B_pj_P = b_f[pj, p]
                    B_P_pj = b_f[p, pj]
                    
                    a_j_row = a_f[j, :]
                    a_j_col = a_f[:, j]
                    
                    # Compute delta vector for all k
                    # delta_vec[k] = a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + a[k,i]*(b[p[k],pj] - b[p[k],pi]) +
                    #                a[j,k]*(b[pi,p[k]] - b[pj,p[k]]) + a[k,j]*(b[p[k],pi] - b[p[k],pj])
                    
                    diff_row = B_pj_P - B_pi_P
                    diff_col = B_P_pj - B_P_pi
                    
                    # Note: for k=i and k=j, the terms are not exactly right due to the swap affecting p[k] itself.
                    # So we compute the vector sum, subtract the incorrect contributions for k=i and k=j,
                    # and add the exact corner terms.
                    
                    delta_vec = (a_i_row * diff_row) + (a_i_col * diff_col) + (a_j_row * (-diff_row)) + (a_j_col * (-diff_col))
                    
                    # Sum over all k
                    delta_sum = np.sum(delta_vec)
                    
                    # Subtract incorrect contributions for k=i and k=j
                    delta_sum -= delta_vec[i]
                    delta_sum -= delta_vec[j]
                    
                    # Add exact delta for the 4 corner terms (i,i), (i,j), (j,i), (j,j)
                    # Old terms: a[i,i]*b[pi,pi] + a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi] + a[j,j]*b[pj,pj]
                    # New terms: a[i,i]*b[pj,pj] + a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj] + a[j,j]*b[pi,pi]
                    
                    delta_corners = (
                        a_f[i, i] * (b_f[pj, pj] - b_f[pi, pi]) +
                        a_f[i, j] * (b_f[pj, pi] - b_f[pi, pj]) +
                        a_f[j, i] * (b_f[pi, pj] - b_f[pj, pi]) +
                        a_f[j, j] * (b_f[pi, pi] - b_f[pj, pj])
                    )
                    
                    total_delta = delta_sum + delta_corners
                    
                    if total_delta < 0:
                        # Accept swap
                        p[i], p[j] = p[j], p[i]
                        cost += total_delta
                        improved = True
                        # Break inner loop to restart outer loop with new p
                        break
                
                if improved:
                    break
            
            if not improved:
                # Local optimum reached
                break
        
        return p, cost
    
    # GRASP-style diverse initializations
    best_p = None
    best_cost = np.inf
    
    # Number of restarts to attempt, adjusted based on time
    # Each local search might take some time, so we limit restarts
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Generate diverse initial solution
        if restart == 0:
            # Best heuristic: rank matching
            heavy = np.argsort(-a_row_sums, kind='stable')
            central = np.argsort(b_row_sums, kind='stable')
            p_init = np.empty(n, dtype=np.int64)
            p_init[heavy] = central
        else:
            # Randomized construction: pick from top-K candidates
            K = 5 + restart  # Increase diversity as we go
            p_init = np.empty(n, dtype=np.int64)
            assigned_locs = set()
            # Sort facilities by flow row sum descending
            order = np.argsort(-a_row_sums, kind='stable')
            # Sort locations by distance row sum ascending
            loc_order = np.argsort(b_row_sums, kind='stable')
            
            for idx, fac in enumerate(order):
                # Pick from top K unassigned locations
                candidates = []
                for loc in loc_order:
                    if loc not in assigned_locs:
                        candidates.append(loc)
                    if len(candidates) >= K:
                        break
                
                # Randomly select from candidates
                chosen = candidates[np.random.randint(len(candidates))]
                p_init[fac] = chosen
                assigned_locs.add(chosen)
        
        # Run local search
        p_opt, cost_opt = vectorized_local_search(p_init, deadline)
        
        if cost_opt < best_cost:
            best_cost = cost_opt
            best_p = p_opt.copy()
    
    # If no restarts completed (shouldn't happen), use the initial heuristic
    if best_p is None:
        heavy = np.argsort(-a_row_sums, kind='stable')
        central = np.argsort(b_row_sums, kind='stable')
        best_p = np.empty(n, dtype=np.int64)
        best_p[heavy] = central
        best_p = vectorized_local_search(best_p, deadline)[0]
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
