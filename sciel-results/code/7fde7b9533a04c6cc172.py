"""Rewritten program: Simulated Annealing with vectorized O(n) move evaluation for QAP."""
import numpy as np
import time
import random


def solve(a, b):
    n = len(a)
    
    # Precompute float versions for speed
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    
    # Initial solution: rank-matching heuristic
    # Match facilities with high flow sums to locations with low distance sums
    flow_sums = af.sum(axis=1)
    dist_sums = bf.sum(axis=1)
    
    heavy = np.argsort(-flow_sums, kind="stable")
    central = np.argsort(dist_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Compute initial cost
    def compute_cost(p):
        return int(np.sum(af * bf[np.ix_(p, p)]))
    
    current_cost = float(compute_cost(p))
    best_cost = current_cost
    best_p = p.copy()
    
    # Simulated Annealing parameters
    start_time = time.time()
    time_limit = 115.0
    
    # Temperature schedule
    T_start = 500000.0
    T_end = 1.0
    cooling_rate = 0.9995
    
    T = T_start
    
    # Precompute for fast delta computation
    # For moving facility i from location u to location v:
    # delta = T1[v] + T2[v] - T1[u] - T2[u] + af[i,i] * (bf[u,u] - bf[v,u] - bf[u,v] + bf[v,v])
    # where T1[v] = sum_j af[i,j] * bf[v, p[j]]
    #       T2[v] = sum_j af[j,i] * bf[p[j], v]
    
    # We'll maintain these incrementally or recompute as needed.
    # For simplicity and correctness, let's recompute the relevant parts each iteration.
    # But that's O(n^2) per move evaluation which is too slow.
    
    # Better approach: maintain row and column sums of the "interaction" matrix.
    # Let M = bf[np.ix_(p, p)]  (the distance matrix under current permutation)
    # Then cost = sum_{i,j} af[i,j] * M[i,j]
    
    # For moving facility i from u to v:
    # The change involves:
    # 1. Terms where i is the first index: sum_j af[i,j] * (bf[v, p[j]] - bf[u, p[j]])
    # 2. Terms where i is the second index: sum_j af[j,i] * (bf[p[j], v] - bf[p[j], u])
    # 3. Diagonal term: af[i,i] * (bf[v,v] - bf[u,u])
    
    # Let's precompute:
    #   R_i[v] = sum_j af[i,j] * bf[v, p[j]]  for all v
    #   C_i[v] = sum_j af[j,i] * bf[p[j], v]  for all v
    # Then delta(i, u, v) = R_i[v] - R_i[u] + C_i[v] - C_i[u] + af[i,i] * (bf[v,v] - bf[u,u])
    
    # But R_i and C_i depend on p, so they change when we move any facility.
    # Maintaining them incrementally is complex.
    
    # Alternative: use a different formulation.
    # cost = sum_i sum_j af[i,j] * bf[p[i], p[j]]
    # Let's define for each facility i and location l:
    #   f(i, l) = sum_j af[i,j] * bf[l, p[j]] + sum_j af[j,i] * bf[p[j], l]
    # Then the contribution of facility i at location p[i] is f(i, p[i]), but this double-counts.
    # Actually, cost = 0.5 * sum_i f(i, p[i]) if af and bf are symmetric, but they're not necessarily.
    
    # Let's just do the direct O(n) computation per candidate move, but vectorize over all target locations.
    
    # For a given facility i currently at location u = p[i]:
    # For each possible new location v (v != u):
    #   delta(v) = sum_j!=i af[i,j]*(bf[v,p[j]] - bf[u,p[j]]) + sum_j!=i af[j,i]*(bf[p[j],v] - bf[p[j],u]) + af[i,i]*(bf[v,v] - bf[u,u])
    #             + af[i,i]*(bf[v,u] - bf[u,u]) + af[i,i]*(bf[u,v] - bf[u,u])  -- wait, need to be careful
    
    # Let me re-derive more carefully.
    # Original cost terms involving i:
    #   sum_j af[i,j] * bf[u, p[j]]  +  sum_j af[j,i] * bf[p[j], u]
    #   This includes j=i: af[i,i]*bf[u,u] + af[i,i]*bf[u,u] = 2*af[i,i]*bf[u,u]
    #   And for j != i: af[i,j]*bf[u,p[j]] + af[j,i]*bf[p[j],u]
    
    # New cost terms involving i (now at v):
    #   sum_j af[i,j] * bf[v, p[j]]  +  sum_j af[j,i] * bf[p[j], v]
    #   This includes j=i: af[i,i]*bf[v,v] + af[i,i]*bf[v,v] = 2*af[i,i]*bf[v,v]
    #   And for j != i: af[i,j]*bf[v,p[j]] + af[j,i]*bf[p[j],v]
    
    # But wait, when we move i from u to v, p changes, so p[j] for j != i stays the same, but the term af[i,j]*bf[?,p[j]] changes from bf[u,p[j]] to bf[v,p[j]], and similarly for the other direction.
    # Also, the term for j where p[j] = u (i.e., j = i since p is a permutation) -- no, p is a permutation so only one facility is at u, which is i. So no other facility is at u.
    
    # So the delta is:
    #   [sum_j af[i,j]*bf[v,p[j]] + sum_j af[j,i]*bf[p[j],v]] - [sum_j af[i,j]*bf[u,p[j]] + sum_j af[j,i]*bf[p[j],u]]
    #   = sum_j af[i,j]*(bf[v,p[j]] - bf[u,p[j]]) + sum_j af[j,i]*(bf[p[j],v] - bf[p[j],u])
    
    # This is correct. The j=i terms are included:
    #   af[i,i]*(bf[v,u] - bf[u,u]) + af[i,i]*(bf[u,v] - bf[u,u])  for the first sum (j=i)
    #   Wait, p[i] = u, so bf[v, p[i]] = bf[v, u] and bf[u, p[i]] = bf[u, u].
    #   Similarly for the second sum: bf[p[i], v] = bf[u, v] and bf[p[i], u] = bf[u, u].
    
    # So delta = sum_j af[i,j]*(bf[v,p[j]] - bf[u,p[j]]) + sum_j af[j,i]*(bf[p[j],v] - bf[p[j],u])
    
    # We can compute this for all v at once using vectorization.
    
    # Let's precompute for the current p:
    #   For each i, we need:
    #     A_i = af[i, :]  (row i of a)
    #     B_i = af[:, i]  (column i of a)
    
    # For a given i and u = p[i]:
    #   term1[v] = sum_j af[i,j] * bf[v, p[j]]  for all v
    #   term2[v] = sum_j af[j,i] * bf[p[j], v]  for all v
    
    # term1[v] = bf[v, :] @ (af[i, :] * 1) ... no.
    # term1[v] = sum_j af[i,j] * bf[v, p[j]]
    # This is like: for each v, sum over j of af[i,j] * bf[v, p[j]].
    # Let M = bf[np.ix_(p, p)]  (n x n matrix where M[i,j] = bf[p[i], p[j]])
    # Then term1[v] = sum_j af[i,j] * bf[v, p[j]]
    # Note that bf[v, p[j]] is not directly M. M[i,j] = bf[p[i], p[j]].
    # So bf[v, p[j]] = M[?, j] where ? is the index k such that p[k] = v. Let inv_p[v] = k.
    # So bf[v, p[j]] = M[inv_p[v], j].
    # Therefore term1[v] = sum_j af[i,j] * M[inv_p[v], j] = af[i, :] @ M[inv_p[v], :].T
    # Similarly, term2[v] = sum_j af[j,i] * bf[p[j], v] = sum_j af[j,i] * M[j, inv_p[v]] = af[:, i] @ M[:, inv_p[v]]
    
    # This requires maintaining M and inv_p, and doing matrix-vector products.
    # For each i, computing term1 and term2 for all v is O(n^2) if done naively.
    # But we only need to do this for one i per iteration, so O(n^2) per iteration.
    # With n=150, that's 22500 ops per iteration, which is fast in NumPy.
    
    # Let's implement this.
    
    M = bf[np.ix_(p, p)]  # M[i,j] = bf[p[i], p[j]]
    
    def get_delta_for_facility(i, u, M, p, af, bf, n):
        """Compute delta for moving facility i from location u to each possible location v."""
        # u = p[i]
        # term1[v] = sum_j af[i,j] * bf[v, p[j]]
        # term2[v] = sum_j af[j,i] * bf[p[j], v]
        
        # bf[v, p[j]] for all v, j: this is bf[:, p] which is bf[:, p] (n x n)
        # Let's compute bf_at_p = bf[:, p]  (n x n, bf_at_p[v, j] = bf[v, p[j]])
        # Then term1[v] = sum_j af[i,j] * bf_at_p[v, j] = (af[i, :] @ bf_at_p[v, :])
        # So term1 = bf_at_p @ af[i, :]  (n-vector)
        
        # bf[p[j], v] for all j, v: this is bf[p, :] which is bf[p, :] (n x n)
        # Let's compute bf_p = bf[p, :]  (n x n, bf_p[j, v] = bf[p[j], v])
        # Then term2[v] = sum_j af[j,i] * bf_p[j, v] = (af[:, i] @ bf_p[:, v])
        # So term2 = bf_p.T @ af[:, i]  (n-vector)
        
        bf_at_p = bf[:, p]  # (n, n): bf_at_p[v, j] = bf[v, p[j]]
        bf_p = bf[p, :]     # (n, n): bf_p[j, v] = bf[p[j], v]
        
        term1 = bf_at_p @ af[i, :]  # (n,)
        term2 = bf_p.T @ af[:, i]   # (n,)
        
        # term1_old = term1[u], term2_old = term2[u]
        # delta[v] = term1[v] - term1[u] + term2[v] - term2[u]
        
        delta = (term1 - term1[u]) + (term2 - term2[u])
        
        # v cannot be u (no change), and v must be a location not occupied by another facility
        # Since p is a permutation, all locations are occupied. Moving i from u to v means
        # the facility currently at v will need to go somewhere else. But in a single-move
        # neighborhood, we just swap i's location with the facility at v.
        # Actually, in standard QAP local search, a "move" is swapping two facilities.
        # So moving i from u to v is equivalent to swapping i and the facility k where p[k] = v.
        # The delta formula above assumes that only i moves, which is not quite right for a swap.
        
        # Let me reconsider. For a swap of facilities i and k (where p[k] = v, p[i] = u):
        # The change in cost is more complex. Let's use the swap delta formula.
        
        # Actually, let's just use the 2-swap (swap two facilities) neighborhood.
        # For swapping facilities i and k:
        # Old: a[i,k]*b[u,v] + a[k,i]*b[v,u] + sum_{j!=i,k} [a[i,j]*b[u,p[j]] + a[j,i]*b[p[j],u] + a[k,j]*b[v,p[j]] + a[j,k]*b[p[j],v]]
        # New: a[i,k]*b[v,u] + a[k,i]*b[u,v] + sum_{j!=i,k} [a[i,j]*b[v,p[j]] + a[j,i]*b[p[j],v] + a[k,j]*b[u,p[j]] + a[j,k]*b[p[j],u]]
        
        # delta = [a[i,k]*b[v,u] + a[k,i]*b[u,v] - a[i,k]*b[u,v] - a[k,i]*b[v,u]]
        #       + sum_{j!=i,k} [a[i,j]*(b[v,p[j]] - b[u,p[j]]) + a[j,i]*(b[p[j],v] - b[p[j],u])
        #                        + a[k,j]*(b[u,p[j]] - b[v,p[j]]) + a[j,k]*(b[p[j],u] - b[p[j],v])]
        
        # This is O(n) per swap. With n=150, there are ~11175 swaps. Total ~1.7M ops per iteration.
        # In Python with NumPy vectorization, this might be feasible if we vectorize over all swaps.
        
        # Let's go back to the single-facility move approach but interpret it correctly.
        # Actually, the formula I derived earlier for moving facility i from u to v (without swapping)
        # is not valid because it doesn't account for the facility that was at v.
        
        # Let's use a different approach: use the "insertion" neighborhood or just do random restarts
        # with a simpler local search.
        
        # Given the time constraints, let's use a simpler approach:
        # 1. Start with rank-matching
        # 2. Do a simple 2-opt local search with random restarts
        # 3. Use simulated annealing with a simple acceptance criterion
        
        # For now, let's just return the initial solution and add a simple local search.
        return delta
    
    # Given the complexity, let's use a simpler but effective approach:
    # Simulated Annealing with random pair swaps and O(n) delta computation.
    
    # Precompute nothing fancy, just do the delta computation on the fly.
    
    T = T_start
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        # Pick a random pair of facilities to swap
        i, k = random.sample(range(n), 2)
        u = p[i]
        v = p[k]
        
        if u == v:
            continue
        
        # Compute delta for swapping i and k
        # delta = [af[i,k]*bf[v,u] + af[k,i]*bf[u,v] - af[i,k]*bf[u,v] - af[k,i]*bf[v,u]]
        #       + sum_{j!=i,k} [af[i,j]*(bf[v,p[j]] - bf[u,p[j]]) + af[j,i]*(bf[p[j],v] - bf[p[j],u])
        #                        + af[k,j]*(bf[u,p[j]] - bf[v,p[j]]) + af[j,k]*(bf[p[j],u] - bf[p[j],v])]
        
        # Vectorize the sum over j
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[k] = False
        js = np.where(mask)[0]
        pj = p[js]
        
        term1 = af[i, js] * (bf[v, pj] - bf[u, pj])
        term2 = af[js, i] * (bf[pj, v] - bf[pj, u])
        term3 = af[k, js] * (bf[u, pj] - bf[v, pj])
        term4 = af[js, k] * (bf[pj, u] - bf[pj, v])
        
        sum_j = np.sum(term1) + np.sum(term2) + np.sum(term3) + np.sum(term4)
        
        diag_delta = af[i, k] * bf[v, u] + af[k, i] * bf[u, v] - af[i, k] * bf[u, v] - af[k, i] * bf[v, u]
        
        delta = diag_delta + sum_j
        
        # Acceptance criterion
        if delta < 0:
            accept = True
        else:
            # Boltzmann acceptance
            if T > 1e-10:
                accept = random.random() < np.exp(-delta / T)
            else:
                accept = False
        
        if accept:
            p[i], p[k] = p[k], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        
        if no_improve_count > max_no_improve:
            # Restart with a perturbed solution
            for _ in range(10):
                i, k = random.sample(range(n), 2)
                p[i], p[k] = p[k], p[i]
            current_cost = float(compute_cost(p))
            T = T_start * 0.5
            no_improve_count = 0
    
    return best_p.tolist()
