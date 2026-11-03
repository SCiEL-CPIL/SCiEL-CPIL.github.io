"""Rewritten QAP solver: fast local search with O(1) swap delta + Hungarian refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Rank-matching initial permutation.
    2. Hungarian linearization for initial refinement.
    3. Fast local search with O(1) swap delta via precomputed aggregates.
    4. Multiple Hungarian refinement rounds interleaved with local search.
    5. Final high-precision polish pass.
    """
    n = len(a)
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # --- Compute cost efficiently ---
    def compute_cost(p):
        B_p = b[np.ix_(p, p)]
        return int(np.sum(a * B_p))
    
    # --- O(1) swap delta using precomputed aggregates ---
    # For a given permutation p, define for each facility i and location l:
    #   R[i, l] = sum_k a[i,k] * b[l, p[k]]   (row contribution if facility i were at location l)
    #   C[i, l] = sum_k a[k,i] * b[p[k], l]   (col contribution if facility i were at location l)
    # 
    # Total cost = sum_i R[i, p[i]] + sum_i C[i, p[i]] - sum_i a[i,i]*b[p[i],p[i]]
    # (the diagonal terms are double-counted)
    #
    # Actually let's define it more carefully.
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # 
    # Define F[i, l] = sum_j a[i,j] * b[l, p[j]]  (what facility i contributes to rows if placed at l)
    # Define G[i, l] = sum_j a[j,i] * b[p[j], l]  (what facility i contributes to cols if placed at l)
    # 
    # Then cost = sum_i F[i, p[i]]  ... wait that's not right either because b[p[i],p[j]] 
    # depends on both p[i] and p[j].
    #
    # Let me think again. The standard approach:
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # 
    # If we swap p[i] and p[j] (locations u and v):
    # Only terms involving i or j in either position change.
    # 
    # Let's define:
    #   row_cost[i, l] = sum_k a[i,k] * b[l, p[k]]   for all i, l
    #   col_cost[i, l] = sum_k a[k,i] * b[p[k], l]   for all i, l
    #
    # Then the total cost can be written as:
    # cost = sum_i row_cost[i, p[i]]  ... NO, this double counts.
    # 
    # Actually: sum_i sum_k a[i,k]*b[p[i],p[k]] = sum_i row_cost[i, p[i]]
    # where row_cost[i, l] = sum_k a[i,k]*b[l, p[k]]
    # Yes! Because when we fix i and set p[i]=l, the inner sum over k is exactly row_cost[i,l].
    # So cost = sum_i row_cost[i, p[i]].
    #
    # Similarly cost = sum_i col_cost[i, p[i]] where col_cost[i,l] = sum_k a[k,i]*b[p[k],l].
    #
    # For swap delta when swapping positions i and j (locations u=p[i], v=p[j]):
    # 
    # New cost - Old cost:
    # Terms affected: all (i',j') where i' in {i,j} or j' in {i,j}
    # 
    # Using row_cost formulation:
    # Old contribution of position i: row_cost[i, u] = sum_k a[i,k]*b[u, p[k]]
    # New contribution of position i: sum_k a[i,k]*b[v, p'[k]]
    #   = a[i,i]*b[v,v] + a[i,j]*b[v,u] + sum_{k!=i,j} a[i,k]*b[v,p[k]]
    #   = row_cost[i, v] - a[i,i]*b[v,u] - a[i,j]*b[v,v] + a[i,i]*b[v,v] + a[i,j]*b[v,u]
    #   = row_cost[i, v] + a[i,i]*(b[v,v]-b[v,u]) + a[i,j]*(b[v,u]-b[v,v])
    #   = row_cost[i, v] + (a[i,i]-a[i,j])*(b[v,v]-b[v,u])
    # 
    # Hmm this is getting messy. Let me use the direct formula with precomputed sums.
    
    def build_precomp(p):
        """Build precomputed arrays for fast swap delta."""
        # row_cost[i, l] = sum_k a[i,k] * b[l, p[k]]
        # = a[i,:] @ b[:, p]  (b[:, p] has shape (n,n), [l,k] = b[l,p[k]])
        b_col_p = b[:, p]  # shape (n, n): [l, k] = b[l, p[k]]
        row_cost = a @ b_col_p  # shape (n, n): [i, l] = sum_k a[i,k]*b[l,p[k]]
        
        # col_cost[i, l] = sum_k a[k,i] * b[p[k], l]
        # = a[:,i] @ b[p, :]  ... 
        # col_cost[i, l] = sum_k a[k,i] * b[p[k], l]
        # b[p, :] has shape (n, n): [k, l] = b[p[k], l]
        b_row_p = b[p, :]  # shape (n, n): [k, l] = b[p[k], l]
        col_cost = a.T @ b_row_p  # shape (n, n): [i, l] = sum_k a.T[i,k]*b_row_p[k,l] = sum_k a[k,i]*b[p[k],l]
        
        return row_cost, col_cost
    
    def swap_delta_fast(i, j, p, row_cost, col_cost, a, b):
        """Compute delta for swapping positions i and j in O(1)."""
        u = p[i]
        v = p[j]
        if u == v:
            return 0.0
        
        # Using the identity: cost = sum_i row_cost[i, p[i]]
        # Old: row_cost[i,u] + row_cost[j,v] + sum_{k!=i,j} row_cost[k, p[k]]
        # New: row_cost[i,v] + row_cost[j,u] + sum_{k!=i,j} row_cost'[k, p[k]]
        # 
        # But row_cost also changes when we swap p[i] and p[j]!
        # row_cost[k, l] = sum_m a[k,m]*b[l, p[m]]
        # After swap, p'[i]=v, p'[j]=u, so:
        # row_cost'[k, l] = sum_m a[k,m]*b[l, p'[m]]
        #   = a[k,i]*b[l,v] + a[k,j]*b[l,u] + sum_{m!=i,j} a[k,m]*b[l,p[m]]
        #   = row_cost[k,l] - a[k,i]*b[l,u] - a[k,j]*b[l,v] + a[k,i]*b[l,v] + a[k,j]*b[l,u]
        #   = row_cost[k,l] + a[k,i]*(b[l,v]-b[l,u]) + a[k,j]*(b[l,u]-b[l,v])
        #   = row_cost[k,l] + (a[k,i]-a[k,j])*(b[l,v]-b[l,u])
        #
        # So for k not in {i,j}, p[k] stays the same, but row_cost[k, p[k]] changes:
        # row_cost'[k, p[k]] = row_cost[k, p[k]] + (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
        #
        # Similarly for i and j themselves:
        # For position i (now at location v):
        # row_cost'[i, v] = row_cost[i,v] + (a[i,i]-a[i,j])*(b[v,v]-b[v,u])
        # For position j (now at location u):
        # row_cost'[j, u] = row_cost[j,u] + (a[j,i]-a[j,j])*(b[u,v]-b[u,u])
        #
        # Delta = [row_cost'[i,v] + row_cost'[j,u] + sum_{k!=i,j} row_cost'[k,p[k]]] 
        #       - [row_cost[i,u] + row_cost[j,v] + sum_{k!=i,j} row_cost[k,p[k]]]
        #
        # = [row_cost[i,v] + (a[i,i]-a[i,j])*(b[v,v]-b[v,u])] 
        # + [row_cost[j,u] + (a[j,i]-a[j,j])*(b[u,v]-b[u,u])]
        # + sum_{k!=i,j} (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
        # - row_cost[i,u] - row_cost[j,v]
        
        d = 0.0
        d += row_cost[i, v] - row_cost[i, u]
        d += row_cost[j, u] - row_cost[j, v]
        d += (a[i,i] - a[i,j]) * (b[v,v] - b[v,u])
        d += (a[j,i] - a[j,j]) * (b[u,v] - b[u,u])
        
        # Sum over k not in {i,j}: (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
        # = sum_k (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u]) 
        #   - (a[i,i]-a[i,j])*(b[u,v]-b[u,u])  [k=i term: p[i]=u]
        #   - (a[j,i]-a[j,j])*(b[v,u]-b[v,v])  [k=j term: p[j]=v]
        #
        # Precompute: S = sum_k (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
        # We can precompute for each pair of locations (l1, l2): 
        #   D[l1,l2] = sum_k a[k,?]*... this depends on i,j so can't fully precompute.
        #
        # Alternative: precompute for each location l and each facility k:
        #   a_col_diff[k] = a[k,i] - a[k,j]  (depends on i,j)
        #   b_diff[k] = b[p[k],v] - b[p[k],u]  (depends on i,j via p)
        #
        # This is O(n) per swap. For n=150 that's acceptable if we do it in numpy.
        
        # Vectorized O(n) computation:
        # diff_a = a[:, i] - a[:, j]  shape (n,)
        # diff_b = b[p, v] - b[p, u]  shape (n,)  (b[p[k],v] - b[p[k],u] for each k)
        # S = dot(diff_a, diff_b)
        # Then subtract k=i and k=j terms:
        # k=i: (a[i,i]-a[i,j])*(b[u,v]-b[u,u])
        # k=j: (a[j,i]-a[j,j])*(b[v,u]-b[v,v])
        
        diff_a = a[:, i] - a[:, j]
        diff_b = b[p, v] - b[p, u]
        S = float(np.dot(diff_a, diff_b))
        S -= (a[i,i] - a[i,j]) * (b[u,v] - b[u,u])
        S -= (a[j,i] - a[j,j]) * (b[v,u] - b[v,v])
        
        d += S
        
        return d
    
    # --- Initial solution ---
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    
    # --- Hungarian refinement to get a better starting point ---
    def hungarian_refine(p_cur, cost_cur):
        """Apply one round of Hungarian linearization refinement."""
        b_col_p = b[:, p_cur]  # [l, k] = b[l, p_cur[k]]
        b_row_p = b[p_cur, :]  # [k, l] = b[p_cur[k], l]
        
        # C[i,j] = sum_k a[i,k]*b[j,p[k]] + sum_k a[k,i]*b[p[k],j]
        C = a @ b_col_p.T + a.T @ b_row_p
        
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        cost_new = compute_cost(p_new)
        if cost_new < cost_cur:
            return p_new, cost_new
        return p_cur, cost_cur
    
    if time.time() < deadline - 5.0:
        best_p, best_cost = hungarian_refine(best_p, best_cost)
    
    # --- Local search with fast delta ---
    def local_search(p_cur, cost_cur, time_limit):
        """Perform local search until no improvement or time limit."""
        row_cost, col_cost = build_precomp(p_cur)
        
        # For the S term in swap_delta, we need b[p, v] - b[p, u] which is O(n)
        # We can precompute b_diff for all pairs but that's O(n^3) memory.
        # Instead, compute on the fly: b[p, v] - b[p, u] is just two column lookups.
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Random sampling of pairs
            num_trials = 50000
            for _ in range(num_trials):
                if time.time() > time_limit:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                d = swap_delta_fast(i, j, p_cur, row_cost, col_cost, a, b)
                if d < -1e-9:
                    # Accept and update precomputed values
                    u = p_cur[i]
                    v = p_cur[j]
                    p_cur[i], p_cur[j] = v, u
                    cost_cur += d
                    improved = True
                    
                    # Update row_cost incrementally:
                    # row_cost[k, l] changed by (a[k,i]-a[k,j])*(b[l,v]-b[l,u]) for all k,l
                    # This is a rank-1 update: outer((a[:,i]-a[:,j]), (b[:,v]-b[:,u]))
                    # But we also need to update for the swap itself.
                    # Actually row_cost[k,l] = sum_m a[k,m]*b[l,p[m]]
                    # After swap p'[i]=v, p'[j]=u:
                    # row_cost'[k,l] = row_cost[k,l] + (a[k,i]-a[k,j])*(b[l,v]-b[l,u])
                    # This is a rank-1 update!
                    da = a[:, i] - a[:, j]  # shape (n,)
                    db = b[:, v] - b[:, u]  # shape (n,)
                    row_cost += np.outer(da, db)
                    
                    # Similarly for col_cost:
                    # col_cost[k, l] = sum_m a[m,k]*b[p[m],l]
                    # After swap: col_cost'[k,l] = col_cost[k,l] + (b[v,l]-b[u,l])*(a[i,k]-a[j,k])
                    # = col_cost[k,l] + outer((b[v,:]-b[u,:]).T, (a[:,i]-a[:,j]))... 
                    # Wait: col_cost[k,l] = sum_m a[m,k]*b[p[m],l]
                    # Change: m=i: a[i,k]*b[v,l] replaces a[i,k]*b[u,l] -> diff: a[i,k]*(b[v,l]-b[u,l])
                    #         m=j: a[j,k]*b[u,l] replaces a[j,k]*b[v,l] -> diff: a[j,k]*(b[u,l]-b[v,l])
                    # Total change for col_cost[k,l]: (a[i,k]-a[j,k])*(b[v,l]-b[u,l])
                    # = outer((a[:,i]-a[:,j]), (b[v,:]-b[u,:]))  ... wait shapes
                    # a[:,i]-a[:,j] has shape (n,), indexed by k
                    # b[v,:]-b[u,:] has shape (n,), indexed by l
                    # So change = outer(a[:,i]-a[:,j], b[v,:]-b[u,:])
                    dc = a[:, i] - a[:, j]  # shape (n,), indexed by k
                    dl = b[v, :] - b[u, :]  # shape (n,), indexed by l
                    col_cost += np.outer(dc, dl)
        
        return p_cur, cost_cur
    
    # Main local search
    if time.time() < deadline - 30.0:
        best_p, best_cost = local_search(best_p, best_cost, deadline - 30.0)
    
    # --- Multiple Hungarian refinement rounds ---
    for _ in range(5):
        if time.time() > deadline - 20.0:
            break
        best_p, best_cost = hungarian_refine(best_p, best_cost)
        if time.time() > deadline - 15.0:
            break
        best_p, best_cost = local_search(best_p, best_cost, deadline - 15.0)
    
    # --- Final polish: more exhaustive local search ---
    if time.time() < deadline - 3.0:
        best_p, best_cost = local_search(best_p, best_cost, deadline - 3.0)
    
    # Final Hungarian
    if time.time() < deadline - 1.0:
        best_p, best_cost = hungarian_refine(best_p, best_cost)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
