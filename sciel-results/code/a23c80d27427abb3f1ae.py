"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Safety margin: 120s limit, aim to finish by 115s
    deadline = time.time() + 115.0

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def swap_delta(i, j, p):
        u = p[i]
        v = p[j]
        
        # Cost = sum_{k,l} a[k,l] * b[p[k], p[l]]
        # After swapping p[i] and p[j]:
        # Delta = sum_l a[i,l]*(b[v,p[l]]-b[u,p[l]]) + sum_l a[l,i]*(b[p[l],v]-b[p[l],u])
        #       - sum_l a[j,l]*(b[u,p[l]]-b[v,p[l]]) - sum_l a[l,j]*(b[p[l],u]-b[p[l],v])
        # But we must exclude l=i and l=j from the sums, then add the direct i-j interaction change.
        
        # Vectorized: compute full sums then correct
        diff_a_rows = a[i, :] - a[j, :]
        diff_b_rows = b[v, p] - b[u, p]
        s1 = np.dot(diff_a_rows, diff_b_rows)
        
        diff_a_cols = a[:, i] - a[:, j]
        diff_b_cols = b[p, v] - b[p, u]
        s2 = np.dot(diff_a_cols, diff_b_cols)
        
        # s1 includes terms for l=i: (a[i,i]-a[j,i])*(b[v,u]-b[u,u])
        # and l=j: (a[i,j]-a[j,j])*(b[v,v]-b[u,v])
        # s2 includes terms for l=i: (a[i,i]-a[j,i])*(b[u,v]-b[u,u])
        # and l=j: (a[i,j]-a[j,j])*(b[u,v]-b[v,v])
        
        # We need to REMOVE these 4 terms from s1+s2, then ADD the correct interaction delta.
        # Correct interaction delta: a[i,j]*b[v,u] + a[j,i]*b[u,v] - a[i,j]*b[u,v] - a[j,i]*b[v,u]
        # = (a[i,j]-a[j,i])*(b[v,u]-b[u,v])
        
        ai_i = a[i, i]
        aj_i = a[j, i]
        ai_j = a[i, j]
        aj_j = a[j, j]
        
        bu_u = b[u, u]
        bu_v = b[u, v]
        bv_u = b[v, u]
        bv_v = b[v, v]
        
        # Terms to subtract (they were included in s1+s2 but should not be)
        term_ii_s1 = (ai_i - aj_i) * (bv_u - bu_u)
        term_ii_s2 = (ai_i - aj_i) * (bu_v - bu_u)
        term_jj_s1 = (ai_j - aj_j) * (bv_v - bu_v)
        term_jj_s2 = (ai_j - aj_j) * (bu_v - bv_v)
        
        # Add correct interaction
        interaction_delta = (ai_j - aj_i) * (bv_u - bu_v)
        
        return s1 + s2 - term_ii_s1 - term_ii_s2 - term_jj_s1 - term_jj_s2 + interaction_delta

    def local_search(p, current_cost, deadline, first_improvement=False):
        """Local search: steepest descent or first improvement.
        first_improvement=True: stop scanning once a negative delta is found and apply it.
        This is much faster per pass, allowing more total passes in the same time budget.
        """
        improved = True
        max_iter = 5000
        iter_count = 0
        time_check_interval = 50

        while improved and time.time() < deadline - 2.0 and iter_count < max_iter:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1

            for i in range(n - 1):
                if iter_count % time_check_interval == 0 and time.time() >= deadline - 2.0:
                    return p, current_cost
                ai = a[i, :]
                ai_col = a[:, i]
                pi = p[i]
                for j in range(i + 1, n):
                    u = p[i]
                    v = p[j]

                    diff_a_rows = ai - a[j, :]
                    diff_b_rows = b[v, p] - b[u, p]
                    s1 = np.dot(diff_a_rows, diff_b_rows)

                    diff_a_cols = ai_col - a[:, j]
                    diff_b_cols = b[p, v] - b[p, u]
                    s2 = np.dot(diff_a_cols, diff_b_cols)

                    ai_i = a[i, i]
                    aj_i = a[j, i]
                    ai_j = a[i, j]
                    aj_j = a[j, j]
                    bu_u = b[u, u]
                    bu_v = b[u, v]
                    bv_u = b[v, u]
                    bv_v = b[v, v]

                    term_ii_s1 = (ai_i - aj_i) * (bv_u - bu_u)
                    term_ii_s2 = (ai_i - aj_i) * (bu_v - bu_u)
                    term_jj_s1 = (ai_j - aj_j) * (bv_v - bu_v)
                    term_jj_s2 = (ai_j - aj_j) * (bu_v - bv_v)
                    interaction_delta = (ai_j - aj_i) * (bv_u - bu_v)

                    d = s1 + s2 - term_ii_s1 - term_ii_s2 - term_jj_s1 - term_jj_s2 + interaction_delta

                    if d < best_delta:
                        best_delta = d
                        best_i = i
                        best_j = j
                        if first_improvement:
                            break
                if first_improvement and best_delta < 0:
                    break

            if best_delta < 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
                iter_count += 1
            else:
                break

        return p, current_cost

    def greedy_init():
        """
        Greedy construction: assign facilities in order of flow sum 
        to the location that minimizes the incremental cost.
        """
        p = np.full(n, -1, dtype=np.int64)
        flow_sums = a.sum(axis=1)
        order = np.argsort(-flow_sums, kind='stable')
        
        assigned = np.zeros(n, dtype=bool)
        
        for idx in order:
            best_loc = -1
            best_inc = np.inf
            
            # For each unassigned location, calculate the incremental cost
            # of placing facility idx there.
            # Incremental cost = sum_{j: assigned} (a[idx, j] + a[j, idx]) * b[loc, p[j]]
            
            # We can vectorize this if we maintain a list of assigned facilities and their locations.
            # Let assigned_facilities be the list of facilities already placed.
            # Let assigned_locations be p[assigned_facilities].
            
            # To keep it simple and fast enough for n=150, we can just loop through unassigned locations
            # and compute the dot product with the already assigned part.
            
            # Optimization: Precompute the interaction vector for facility idx with all other facilities
            # inter_vec = a[idx, :] + a[:, idx]
            # Then for a location loc, inc = sum_{j in assigned} inter_vec[j] * b[loc, p[j]]
            
            inter_vec = a[idx, :] + a[:, idx]
            
            # Get indices of assigned facilities
            assigned_mask = assigned
            # We need to sum inter_vec[j] * b[loc, p[j]] for all j where assigned[j] is True
            
            # b[loc, p] is a vector of length n.
            # We can multiply inter_vec by a mask of assigned, then dot with b[loc, p]?
            # No, b[loc, p[j]] depends on j.
            
            # Let's just loop over unassigned locations. n is 150.
            # Inner loop is over assigned facilities (up to 150).
            # Total ops ~ n^3 / 2 = 1.7M. Very fast.
            
            for loc in range(n):
                if assigned[loc]:
                    continue
                
                # Calculate incremental cost
                # Sum over all j such that assigned[j] is True
                # term_j = inter_vec[j] * b[loc, p[j]]
                
                # Vectorized way:
                # mask = assigned.astype(float)
                # terms = inter_vec * mask
                # inc = np.dot(terms, b[loc, p]) # This is wrong because p[j] is not j.
                
                # Correct vectorized way:
                # We have b[loc, :] which is a row of B.
                # We want sum_{j: assigned} inter_vec[j] * b[loc, p[j]]
                # This is equivalent to:
                # Create a vector v where v[p[j]] = inter_vec[j] if assigned[j] else 0?
                # That's complicated.
                
                # Let's just use the assigned indices.
                assigned_indices = np.where(assigned)[0]
                if len(assigned_indices) == 0:
                    inc = 0.0
                else:
                    # b[loc, p[assigned_indices]] gives the distances from loc to the locations of assigned facilities
                    # inter_vec[assigned_indices] gives the flow interactions
                    inc = np.dot(inter_vec[assigned_indices], b[loc, p[assigned_indices]])
                
                if inc < best_inc:
                    best_inc = inc
                    best_loc = loc
            
            p[idx] = best_loc
            assigned[best_loc] = True
            
        return p

    # 1. Initialization
    best_cost = np.inf
    best_p = None

    # Generate diverse initial solutions
    inits = []

    # Greedy Init
    inits.append(greedy_init())

    # Rank Init (Forward)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    inits.append(p_rank)

    # Rank Init (Reversed)
    p_rank_rev = np.empty(n, dtype=np.int64)
    p_rank_rev[heavy] = central[::-1]
    inits.append(p_rank_rev)

    # Random Init
    rng = np.random.RandomState(42)
    inits.append(rng.permutation(n))

    # Run local search from each initialization (use first-improvement for speed)
    for p_init in inits:
        if time.time() > deadline - 10.0:
            break

        p_work = p_init.copy()
        cost = compute_cost(p_work)

        p_work, cost = local_search(p_work, cost, deadline, first_improvement=True)

        if cost < best_cost:
            best_cost = cost
            best_p = p_work.copy()

    if best_p is None:
        best_p = inits[0].copy()
        best_cost = compute_cost(best_p)

    # 2. Iterated Local Search (ILS)
    # Use random perturbations to escape local minima
    no_improve_count = 0
    rng = np.random.RandomState(123)

    while time.time() < deadline - 8.0:
        # Perturb the best solution
        p = best_p.copy()

        # Adaptive perturbation: use 3-opt style moves for better exploration
        base_swaps = 3
        extra_swaps = min(no_improve_count, 10)
        num_swaps = base_swaps + extra_swaps
        num_swaps = min(num_swaps, 15)

        for _ in range(num_swaps):
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i != j:
                p[i], p[j] = p[j], p[i]

        current_cost = compute_cost(p)

        # Use first-improvement LS for ILS to allow more restarts
        p, current_cost = local_search(p, current_cost, deadline, first_improvement=True)

        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count > 20:
            no_improve_count = 0

    # 3. Final high-precision polish: steepest-descent local search with more iterations
    # This is the "tighter tolerance, more iterations" polish pass.
    if best_p is not None and time.time() < deadline - 3.0:
        p_polish = best_p.copy()
        cost_polish = compute_cost(p_polish)
        # Run steepest-descent local search with the remaining time
        p_polish, cost_polish = local_search(p_polish, cost_polish, deadline, first_improvement=False)
        if cost_polish < best_cost:
            best_cost = cost_polish
            best_p = p_polish

    # Final verification
    if best_p is not None:
        final_cost = compute_cost(best_p)
        if final_cost < best_cost:
            best_cost = final_cost
        return best_p.tolist()

    return list(range(n))

# EVOLVE-BLOCK-END
