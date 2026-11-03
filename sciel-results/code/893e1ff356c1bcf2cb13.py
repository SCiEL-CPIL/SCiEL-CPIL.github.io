"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def calc_cost(p):
        """Calculate the QAP cost for a permutation p."""
        # p[i] is the location of facility i.
        # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # This is equivalent to sum(a * b[np.ix_(p, p)])
        return int(np.sum(a * b[np.ix_(p, p)]))

    def fast_2opt(p, a, b, time_limit):
        """
        Perform Simulated Annealing with vectorized 2-opt moves.
        p: current permutation (numpy array)
        """
        current_p = p.copy()
        current_cost = calc_cost(current_p)
        best_p = current_p.copy()
        best_cost = current_cost
        
        # SA Parameters
        # Initial temperature needs to be high enough to accept bad moves
        # Estimate typical delta: random swap cost change
        # We can estimate scale from matrix norms or just start high.
        T = 1000.0 
        T_min = 1.0
        cooling = 0.995
        
        # Precompute row sums of A? No, we need full delta.
        # Delta for swapping i and j:
        # Let u = p[i], v = p[j].
        # Old terms involving i or j:
        # sum_k a[i,k] b[u, p[k]] + sum_k a[j,k] b[v, p[k]] 
        # + sum_k a[k,i] b[p[k], u] + sum_k a[k,j] b[p[k], v]
        # (Note: term k=i and k=j are included in these sums, but we must be careful with double counting or self-interactions if i==j, but i!=j here)
        
        # Actually, it is safer to compute the delta explicitly for a candidate pair or vectorize over all j for a fixed i.
        # For a fixed i, let u = p[i].
        # Delta(j) for swapping i and j (v = p[j]):
        # The change in cost comes from interactions of i and j with all other nodes k, 
        # plus the interaction between i and j itself.
        
        # Let's derive the vectorized update for a fixed i.
        # Cost = sum_{x,y} a[x,y] b[p[x], p[y]]
        # Let S be the set of nodes.
        # When swapping p[i] and p[j], only terms where x=i, x=j, y=i, or y=j change.
        # Terms:
        # 1. a[i,i] b[u,u] (unchanged if i==i? No, p[i] changes to v, so b[v,v])
        #    Wait, a[i,i] is fixed. b[p[i], p[i]] becomes b[v,v].
        # 2. a[j,j] b[v,v] becomes b[u,u].
        # 3. a[i,j] b[u,v] becomes a[i,j] b[v,u] = a[i,j] b[u,v] (since b is symmetric). UNCHANGED.
        # 4. a[j,i] b[v,u] becomes a[j,i] b[u,v] = a[j,i] b[v,u] (since a is symmetric). UNCHANGED.
        #    So the interaction between i and j does NOT change cost?
        #    Let's check:
        #    Old: a[i,j] b[u,v] + a[j,i] b[v,u] = 2 * a[i,j] b[u,v] (if symmetric)
        #    New: a[i,j] b[v,u] + a[j,i] b[u,v] = 2 * a[i,j] b[v,u] = 2 * a[i,j] b[u,v].
        #    Yes, the i-j interaction is invariant under swap if A and B are symmetric.
        
        # So we only need to consider interactions with k != i, k != j.
        # Delta = 
        #   sum_{k != i,j} [ a[i,k] (b[v, p[k]] - b[u, p[k]]) + a[k,i] (b[p[k], v] - b[p[k], u]) ]
        # + sum_{k != i,j} [ a[j,k] (b[u, p[k]] - b[v, p[k]]) + a[k,j] (b[p[k], u] - b[p[k], v)) ]
        # + (a[i,i] (b[v,v] - b[u,u]) + a[j,j] (b[u,u] - b[v,v]))
        
        # Since A and B are symmetric:
        # a[k,i] = a[i,k], b[p[k], v] = b[v, p[k]].
        # So for a specific k:
        # Term from i: a[i,k] (b[v, p[k]] - b[u, p[k]]) + a[i,k] (b[v, p[k]] - b[u, p[k]]) = 2 * a[i,k] (b[v, p[k]] - b[u, p[k]])
        # Term from j: a[j,k] (b[u, p[k]] - b[v, p[k]]) + a[j,k] (b[u, p[k]] - b[v, p[k]]) = 2 * a[j,k] (b[u, p[k]] - b[v, p[k]])
        
        # Total Delta(k) for k != i, j:
        # 2 * (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
        
        # Plus self terms:
        # a[i,i] (b[v,v] - b[u,u]) + a[j,j] (b[u,u] - b[v,v]) = (a[i,i] - a[j,j]) * (b[v,v] - b[u,u])
        
        # So, for a fixed i and candidate j:
        # Let u = p[i], v = p[j].
        # We need sum_{k != i, j} 2 * (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
        # + (a[i,i] - a[j,j]) * (b[v,v] - b[u,u])
        
        # This is still O(n) per j. To vectorize over ALL j for a fixed i:
        # We can precompute vectors.
        # Let diff_a = a[i] - a[j]  (vector of length n)
        # Let diff_b = b[v] - b[u]  (vector of length n)
        # We need dot product of diff_a and diff_b, but excluding indices i and j.
        # sum_{k} diff_a[k] * diff_b[p[k]] is not quite right because diff_b is indexed by location, 
        # but the term is b[v, p[k]] - b[u, p[k]].
        # Let's look at the structure: b[v, p[k]] is the k-th element of the row v of B, permuted by p? No.
        # b[v, p[k]] is B[v, p[k]].
        
        # Let's define a vector D_k for each k? No, that's O(n^2) memory.
        # Let's try to compute the sum S = sum_{k} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
        # This is sum_k (a[i,k] - a[j,k]) * b[v, p[k]] - sum_k (a[i,k] - a[j,k]) * b[u, p[k]]
        
        # Let vec_i_j = a[i] - a[j].
        # Let row_v = b[v], row_u = b[u].
        # We need sum_k vec_i_j[k] * row_v[p[k]] and sum_k vec_i_j[k] * row_u[p[k]].
        
        # This is equivalent to:
        # (vec_i_j * (row_v[p]))? No, row_v[p] is an array where index k is row_v[p[k]].
        # So yes, if we construct the array Bp = b[:, p] (columns of b permuted by p), 
        # then Bp[v, k] = b[v, p[k]].
        # So sum_k vec_i_j[k] * Bp[v, k] = dot(vec_i_j, Bp[v])
        # And sum_k vec_i_j[k] * Bp[u, k] = dot(vec_i_j, Bp[u])
        
        # So Delta_approx = 2 * (dot(vec_i_j, Bp[v]) - dot(vec_i_j, Bp[u]))
        # But we must exclude k=i and k=j from the sum.
        # The dot product includes k=i and k=j.
        # Let's subtract the excluded terms:
        # Term k=i: vec_i_j[i] * (Bp[v, i] - Bp[u, i])
        #   vec_i_j[i] = a[i,i] - a[j,i]
        #   Bp[v, i] = b[v, p[i]] = b[v, u]
        #   Bp[u, i] = b[u, p[i]] = b[u, u]
        # Term k=j: vec_i_j[j] * (Bp[v, j] - Bp[u, j])
        #   vec_i_j[j] = a[i,j] - a[j,j]
        #   Bp[v, j] = b[v, p[j]] = b[v, v]
        #   Bp[u, j] = b[u, p[j]] = b[u, v]
        
        # So Delta = 2 * [ (dot(vec_i_j, Bp[v]) - dot(vec_i_j, Bp[u])) 
        #                  - (a[i,i]-a[j,i])*(b[v,u]-b[u,u]) 
        #                  - (a[i,j]-a[j,j])*(b[v,v]-b[u,v]) ]
        #                  + (a[i,i] - a[j,j]) * (b[v,v] - b[u,u])  # Self terms derived earlier
        
        # Let's simplify the self term addition.
        # The formula 2 * sum_{k != i,j} ... + self_term.
        # The dot product calculation effectively calculated 2 * sum_{all k} ... 
        # So we subtract 2 * (excluded terms) and add the correct self_term.
        
        # Let's implement this vectorized check for a fixed i against all j.
        
        Bp = b[:, current_p] # n x n matrix. Bp[loc, fac] = b[loc, p[fac]]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Randomly select an i to propose swaps with
            i = np.random.randint(n)
            u = current_p[i]
            
            # Vectorized calculation of deltas for all j != i
            # vec_i_j for all j?
            # a[i] is fixed. a[j] varies.
            # We need dot(a[i] - a[j], Bp[v] - Bp[u]) for all j.
            # This is hard to vectorize directly because v = p[j] varies for each j.
            # Bp[v] is the row v of Bp.
            
            # Alternative: Just do a random subset of j's or a full scan if n is small enough.
            # n=150. Full scan of j for a fixed i is 150 iterations.
            # Each iteration is O(1) if we precompute? No, the dot product is O(n).
            # So full scan for one i is O(n^2) = 22,500 ops.
            # 1000 iterations of SA -> 22.5M ops. Very fast in NumPy.
            
            # Let's try a full scan for the best move (greedy) or random sampling for SA.
            # For SA, we pick a random j.
            
            # To speed up SA, we can pick a random j and compute delta.
            # But computing delta requires O(n) work.
            # Let's just pick a random j.
            
            j = np.random.randint(n)
            if i == j:
                continue
                
            v = current_p[j]
            
            # Compute Delta
            vec_diff = a[i] - a[j]
            row_v = Bp[v]
            row_u = Bp[u]
            
            # Full dot product
            dp_v = np.dot(vec_diff, row_v)
            dp_u = np.dot(vec_diff, row_u)
            
            # Subtract excluded terms k=i, k=j
            # Term i: (a[i,i] - a[j,i]) * (b[v,u] - b[u,u])
            # Term j: (a[i,j] - a[j,j]) * (b[v,v] - b[u,v])
            excl_i = (a[i,i] - a[j,i]) * (b[v,u] - b[u,u])
            excl_j = (a[i,j] - a[j,j]) * (b[v,v] - b[u,v])
            
            sum_k = dp_v - dp_u - excl_i - excl_j
            
            # Self term: (a[i,i] - a[j,j]) * (b[v,v] - b[u,u])
            self_term = (a[i,i] - a[j,j]) * (b[v,v] - b[u,u])
            
            delta = 2 * sum_k + self_term
            
            # Acceptance
            if delta < 0:
                accept = True
            else:
                # Metropolis criterion
                if T > T_min:
                    accept = np.random.rand() < np.exp(-delta / T)
                else:
                    accept = False
                    
            if accept:
                # Swap
                current_p[i], current_p[j] = v, u
                current_cost += int(delta)
                
                # Update Bp for the swapped columns i and j
                # Bp[:, i] was row u, now should be row v
                # Bp[:, j] was row v, now should be row u
                Bp[:, i] = b[:, v]
                Bp[:, j] = b[:, u]
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
                
                improved = True # We made a move, so loop continues
            else:
                # In SA, we usually don't break on rejection, we just continue.
                # But to ensure we don't infinite loop on rejections, we count rejections?
                # Or just rely on time limit.
                pass
            
            # Cooling
            T *= cooling
            
            # If T is very low, we can switch to greedy (hill climbing) to polish
            if T < T_min:
                # Greedy mode: find best move?
                # For simplicity, just stop or continue with T_min
                T = T_min

        return best_p, best_cost

    # 1. Initialization
    # Use spectral initialization to get a good starting point
    try:
        w_a, v_a = np.linalg.eigh(a)
        w_b, v_b = np.linalg.eigh(b)
        
        # Use top eigenvectors (largest magnitude)
        idx_a = np.argmax(np.abs(w_a))
        idx_b = np.argmax(np.abs(w_b))
        vec_a = v_a[:, idx_a]
        vec_b = v_b[:, idx_b]
        
        # Create assignment based on ranks of projections
        # Project facilities onto vec_a, locations onto vec_b
        # Sort facilities by projection on vec_a
        # Sort locations by projection on vec_b
        # Match them
        
        proj_a = np.dot(vec_a, a) # Not quite, vec_a is in facility space?
        # Actually, eigh returns eigenvectors in the same space as the matrix indices.
        # a[i,j] is flow between i and j.
        # vec_a[i] is the coordinate of facility i in the principal component.
        # vec_b[j] is the coordinate of location j.
        
        # Sort facilities by vec_a
        order_f = np.argsort(vec_a)
        # Sort locations by vec_b
        order_l = np.argsort(vec_b)
        
        # p[i] should be the location matched to facility i.
        # If facility i is at rank k in order_f, it should match location at rank k in order_l.
        p_init = np.empty(n, dtype=np.int64)
        # For each facility i, find its rank
        ranks_f = np.empty(n, dtype=np.int64)
        ranks_f[order_f] = np.arange(n)
        
        # For each location j, find its rank
        ranks_l = np.empty(n, dtype=np.int64)
        ranks_l[order_l] = np.arange(n)
        
        # Match facility i (rank r) to location j (rank r)
        # We need p[i] = j such that ranks_l[j] == ranks_f[i]
        # So j = order_l[ranks_f[i]]
        p_init = order_l[ranks_f]
        
    except Exception:
        p_init = np.random.permutation(n)
    
    # 2. Local Search / SA
    best_p, best_cost = fast_2opt(p_init, a, b, deadline)
    
    # 3. If time remains, try a few more restarts or perturbations?
    # Given the budget, one long SA run is usually better than many short ones.
    # But if we have time, we can try a random restart from a perturbed best_p.
    
    if time.time() < deadline - 10.0:
        # Perturb best_p
        p_pert = best_p.copy()
        # Swap 5 random pairs
        for _ in range(5):
            i, j = np.random.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        p2, c2 = fast_2opt(p_pert, a, b, deadline)
        if c2 < best_cost:
            best_p = p2
            best_cost = c2
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
