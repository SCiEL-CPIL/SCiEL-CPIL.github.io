"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _current_cost(p, a, b):
    """Compute the QAP cost for permutation p: sum a[i,j] * b[p[i],p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def solve(a, b):
    n = len(a)
    t_start = time.time()
    time_budget = 110.0  # leave 10s safety margin

    # --- 1. Greedy construction ---
    # Build permutation by placing facilities one at a time, choosing the location
    # that minimizes incremental cost.
    # For facility i placed at location loc, the incremental cost contribution is:
    #   sum_{j already placed} a[i,j]*b[loc, p[j]] + sum_{j already placed} a[j,i]*b[p[j], loc]
    # We'll maintain the assignment incrementally.

    p = np.full(n, -1, dtype=np.int64)
    # To compute incremental cost efficiently, we track for each candidate location loc,
    # the accumulated cost of placing a new facility at loc given current assignments.
    # inc_cost[loc] = sum over already-placed j of a[i,j]*b[loc,p[j]] + a[j,i]*b[p[j],loc]
    # But this depends on i, so we compute per i.

    # Precompute nothing; compute incrementally.
    # For speed, we'll compute the vector of incremental costs for all unassigned locations
    # for the next facility to place.

    placed = np.zeros(n, dtype=bool)
    assigned_locs = np.zeros(n, dtype=np.int64)  # p for placed facilities

    for i in range(n):
        if time.time() - t_start > time_budget - 5:
            # Fallback: assign remaining to remaining locations
            remaining_facs = np.where(~placed)[0]
            remaining_locs = np.where(np.full(n, True, dtype=bool))[0]
            remaining_locs = [loc for loc in range(n) if not placed[loc]]
            for k, fac in enumerate(remaining_facs):
                if k < len(remaining_locs):
                    p[fac] = remaining_locs[k]
                else:
                    # Should not happen
                    p[fac] = fac
            break

        # Compute incremental cost for each candidate location
        # For facility i at location loc:
        # cost_inc[loc] = sum_{j: placed[j]} a[i,j]*b[loc, p[j]] + a[j,i]*b[p[j], loc]
        # = sum_j a[i,j]*b[loc, p[j]] + sum_j a[j,i]*b[p[j], loc]
        # Note: a and b may not be symmetric, but QAP cost is sum_{i,j} a[i,j]*b[p[i],p[j]]
        # So the term for pair (i,j) is a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]]
        # When we place i at loc, the new terms involving i are:
        #   for each already-placed j: a[i,j]*b[loc, p[j]] + a[j,i]*b[p[j], loc]

        # Vectorized computation over all locations:
        # Let P be the current assignment (p for placed, 0 for unplaced - but we mask)
        # b[loc, p[j]] for all loc and all placed j: this is b[:, p[placed]] * a[i, placed]
        # Actually: sum_j a[i,j] * b[loc, p[j]] = (a[i, placed] * b[loc, p[placed]]).sum() over j
        # Similarly: sum_j a[j,i] * b[p[j], loc] = (a[placed, i] * b[p[placed], loc]).sum() over j

        placed_idx = np.where(placed)[0]
        if len(placed_idx) == 0:
            # First facility: any location is equally good, pick 0
            p[i] = 0
            placed[i] = True
            continue

        # b_at_placed: shape (n, len(placed_idx)) -> b[loc, p[j]] for all loc, j
        # Actually b is (n,n), p[placed_idx] is the locations.
        # b[loc, p[j]] for all loc (rows) and j (cols): b[:, p[placed_idx]]
        # Shape: (n, len(placed_idx))
        b_loc_pj = b[:, p[placed_idx]]  # (n, m) where m = len(placed_idx)
        b_pj_loc = b[p[placed_idx], :]  # (m, n)

        # a_i: a[i, placed_idx] shape (m,)
        a_i = a[i, placed_idx]  # (m,)
        a_ji = a[placed_idx, i]  # (m,)

        # inc1[loc] = sum_j a[i,j] * b[loc, p[j]] = a_i @ b_loc_pj[loc, :] 
        # = (b_loc_pj * a_i[None, :]).sum(axis=1)
        inc1 = (b_loc_pj * a_i[None, :]).sum(axis=1)  # (n,)

        # inc2[loc] = sum_j a[j,i] * b[p[j], loc] = (b_pj_loc * a_ji[:, None]).sum(axis=0)
        inc2 = (b_pj_loc * a_ji[:, None]).sum(axis=0)  # (n,)

        inc_total = inc1 + inc2  # (n,)

        # Mask out already-used locations
        used_mask = placed.copy()
        inc_total[used_mask] = np.inf

        best_loc = np.argmin(inc_total)
        p[i] = best_loc
        placed[i] = True

    # Ensure p is a valid permutation (safety)
    if not np.all(np.sort(p) == np.arange(n)):
        # Fallback to identity-like
        remaining = list(set(range(n)) - set(p.tolist()))
        for i in range(n):
            if p[i] == -1 or p[i] >= n or placed.count(p[i]) > 1:
                # Simple repair
                pass
        # Just reassign: use a simple rank matching as fallback
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central

    # --- 2. 2-opt local search ---
    # Swap p[i] and p[j] for all pairs, keep best improvement.
    # Delta for swapping i and j (i<j):
    # Original terms involving i or j:
    #   a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]]
    #   + sum_{k!=i,j} a[i,k]*b[p[i],p[k]] + a[k,i]*b[p[k],p[i]]
    #   + sum_{k!=i,j} a[j,k]*b[p[j],p[k]] + a[k,j]*b[p[k],p[j]]
    # After swap (p[i]<->p[j]):
    #   a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[i],p[j]]
    #   + sum_{k!=i,j} a[i,k]*b[p[j],p[k]] + a[k,i]*b[p[k],p[j]]
    #   + sum_{k!=i,j} a[j,k]*b[p[i],p[k]] + a[k,j]*b[p[k],p[i]]
    # Delta = (new - old)

    # Vectorized: for a fixed i, compute delta for all j != i.
    # We'll do this in a loop over i, vectorized over j.

    current_cost = _current_cost(p, a, b)
    improved = True
    while improved:
        improved = False
        if time.time() - t_start > time_budget:
            break

        # For each i, compute delta for swapping i with each j
        # delta_ij = 
        #   [a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[i],p[j]]] - [a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]]]
        #   + sum_{k!=i,j} [a[i,k]*b[p[j],p[k]] + a[k,i]*b[p[k],p[j]] + a[j,k]*b[p[i],p[k]] + a[k,j]*b[p[k],p[i]]]
        #   - sum_{k!=i,j} [a[i,k]*b[p[i],p[k]] + a[k,i]*b[p[k],p[i]] + a[j,k]*b[p[j],p[k]] + a[k,j]*b[p[k],p[j]]]

        # Let pi = p[i], pj = p[j]
        # Group 1 (i-j pair): 
        #   delta_pair = a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj] - a[i,j]*b[pi,pj] - a[j,i]*b[pj,pi]
        #              = (a[i,j]-a[j,i])*b[pj,pi] - (a[i,j]-a[j,i])*b[pi,pj]
        #              = (a[i,j]-a[j,i])*(b[pj,pi] - b[pi,pj])

        # Group 2 (i with others k != i,j):
        #   sum_k [a[i,k]*b[pj,p[k]] + a[k,i]*b[p[k],pj] - a[i,k]*b[pi,p[k]] - a[k,i]*b[p[k],pi]]
        #   = sum_k [a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + a[k,i]*(b[p[k],pj] - b[p[k],pi])]

        # Group 3 (j with others k != i,j):
        #   sum_k [a[j,k]*b[pi,p[k]] + a[k,j]*b[p[k],pi] - a[j,k]*b[pj,p[k]] - a[k,j]*b[p[k],pj]]
        #   = sum_k [a[j,k]*(b[pi,p[k]] - b[pj,p[k]]) + a[k,j]*(b[p[k],pi] - b[p[k],pj])]

        # For fixed i, we want delta for all j != i. This is complex to fully vectorize
        # because the sums over k exclude i and j. Let's compute it more carefully.

        # Alternative: compute the total contribution of facility i and j to the cost,
        # then compute it after swap.

        # Contribution of facility i to total cost (all terms involving i as either index):
        #   C_i = sum_j a[i,j]*b[pi,p[j]] + sum_j a[j,i]*b[p[j],pi]
        #        = (a[i,:] * b[pi, p]).sum() + (a[:,i] * b[p, pi]).sum()
        # But this double-counts the i,j term? No, in QAP cost sum_{u,v} a[u,v]*b[p[u],p[v]],
        # the term for (i,j) is a[i,j]*b[pi,pj]. It's counted once.
        # C_i = sum_{v} a[i,v]*b[pi,p[v]] + sum_{u} a[u,i]*b[p[u],pi]
        #      = (a[i,:] * b[pi, p]).sum() + (a[:,i] * b[p, pi]).sum()
        # This includes the (i,i) term: a[i,i]*b[pi,pi] twice? Let's check:
        # In the first sum, v=i gives a[i,i]*b[pi,pi]. In the second sum, u=i gives a[i,i]*b[pi,pi].
        # But in the total cost, the (i,i) term is a[i,i]*b[pi,pi] (once). So C_i counts it twice.
        # Similarly for other terms: the (i,j) term appears in first sum (v=j) and in C_j's second sum (u=i).
        # So total cost = (1/2) * sum_i C_i? No, that's not right either.

        # Let me just compute directly:
        # Total cost = sum_{u,v} a[u,v]*b[p[u],p[v]]
        # = sum_u sum_v a[u,v]*b[p[u],p[v]]

        # For a swap of pi and pj (facilities i and j), the change is:
        # Only terms where u in {i,j} or v in {i,j} change.
        # Specifically:
        # Old terms with i: sum_v a[i,v]*b[pi,p[v]] + sum_u a[u,i]*b[p[u],pi] - a[i,i]*b[pi,pi] (to avoid double count in the i-i term which is in both sums)
        # This is getting complicated. Let me use a different approach.

        # Compute delta by:
        # 1. Save old pi, pj
        # 2. Swap
        # 3. Compute new cost
        # 4. Delta = new - old
        # 5. If delta < 0, keep swap, else revert
        # But computing full cost each time is O(n^2) = 65K, and we have ~32K pairs, so 2B ops per pass. Too slow.

        # Better: use the formula. Let me derive it properly.
        # Let S = sum_{u,v} a[u,v]*b[p[u],p[v]]
        # After swapping p[i] and p[j] (pi <-> pj):
        # The only terms that change are those where u=i or v=i or u=j or v=j.
        # Let's categorize:
        # Terms where u=i, v=i: a[i,i]*b[pi,pi] -> a[i,i]*b[pj,pj]  (wait, no, p[i] becomes pj, so b[p[i],p[i]] = b[pj,pj])
        # Hmm, let me think again. p[i] is the location of facility i. After swap, facility i is at pj and facility j is at pi.
        # So p'[i] = pj, p'[j] = pi, p'[k] = p[k] for k != i,j.
        # 
        # Terms involving i or j in the cost sum:
        # (i,i): a[i,i]*b[p'[i],p'[i]] = a[i,i]*b[pj,pj]  (was a[i,i]*b[pi,pi])
        # (i,j): a[i,j]*b[p'[i],p'[j]] = a[i,j]*b[pj,pi]  (was a[i,j]*b[pi,pj])
        # (j,i): a[j,i]*b[p'[j],p'[i]] = a[j,i]*b[pi,pj]  (was a[j,i]*b[pj,pi])
        # (j,j): a[j,j]*b[p'[j],p'[j]] = a[j,j]*b[pi,pi]  (was a[j,j]*b[pj,pj])
        # (i,k) for k!=i,j: a[i,k]*b[p'[i],p'[k]] = a[i,k]*b[pj,p[k]]  (was a[i,k]*b[pi,p[k]])
        # (k,i) for k!=i,j: a[k,i]*b[p'[k],p'[i]] = a[k,i]*b[p[k],pj]  (was a[k,i]*b[p[k],pi])
        # (j,k) for k!=i,j: a[j,k]*b[p'[j],p'[k]] = a[j,k]*b[pi,p[k]]  (was a[j,k]*b[pj,p[k]])
        # (k,j) for k!=i,j: a[k,j]*b[p'[k],p'[j]] = a[k,j]*b[p[k],pi]  (was a[k,j]*b[p[k],pj])

        # Delta = [a[i,i]*(b[pj,pj]-b[pi,pi]) + a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])]
        #       + sum_{k!=i,j} [a[i,k]*(b[pj,p[k]]-b[pi,p[k]]) + a[k,i]*(b[p[k],pj]-b[p[k],pi])
        #                        + a[j,k]*(b[pi,p[k]]-b[pj,p[k]]) + a[k,j]*(b[p[k],pi]-b[p[k],pj])]

        # Let's define for fixed i, pi = p[i]:
        # For each j != i, with pj = p[j]:
        # delta_ij = 
        #   a[i,i]*(b[pj,pj]-b[pi,pi]) + a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
        #   + sum_{k!=i,j} [a[i,k]*(b[pj,p[k]]-b[pi,p[k]]) + a[k,i]*(b[p[k],pj]-b[p[k],pi])
        #                    + a[j,k]*(b[pi,p[k]]-b[pj,p[k]]) + a[k,j]*(b[p[k],pi]-b[p[k],pj])]

        # This is still complex to vectorize over j for fixed i because of the sum over k!=i,j.
        # Let's compute the sum over all k first, then subtract k=i and k=j terms.

        # For fixed i:
        # Sum1_j = sum_k a[i,k]*(b[pj,p[k]]-b[pi,p[k]])  for all k, then subtract k=i and k=j
        # Sum2_j = sum_k a[k,i]*(b[p[k],pj]-b[p[k],pi])  for all k, then subtract k=i and k=j
        # Sum3_j = sum_k a[j,k]*(b[pi,p[k]]-b[pj,p[k]])  for all k, then subtract k=i and k=j
        # Sum4_j = sum_k a[k,j]*(b[p[k],pi]-b[p[k],pj])  for all k, then subtract k=i and k=j

        # This is getting very complex. Given time constraints, let me use a simpler approach:
        # For each pair (i,j), compute delta by directly computing the affected terms.
        # With n=256, there are ~32K pairs. For each pair, O(n) work = 32K*256 = 8M ops. 
        # With NumPy, this might be feasible if we vectorize over j for each i.

        # Let me try a different vectorization strategy:
        # For fixed i, compute delta for all j simultaneously.

        for i in range(n):
            if time.time() - t_start > time_budget:
                break
            pi = p[i]
            # For all j, compute delta_ij
            # Let's compute the four sums.
            
            # Precompute for all locations loc:
            # b_pi_p[loc] = b[pi, p[loc]]  -- no, we need b[pj, p[k]] etc.
            
            # Let me compute:
            # For each candidate pj (which is p[j] for some j), and for each k:
            # We need b[pj, p[k]] and b[pi, p[k]], etc.
            
            # b[:, p] gives a matrix M where M[loc, k] = b[loc, p[k]]. Shape (n, n).
            # So M[pj, k] = b[pj, p[k]], M[pi, k] = b[pi, p[k]].
            
            M = b[:, p]  # M[loc, k] = b[loc, p[k]], shape (n, n)
            
            # For fixed i, pi:
            # row_pi = M[pi, :]  # b[pi, p[k]] for all k, shape (n,)
            row_pi = M[pi, :]
            
            # For all j, pj = p[j], so we need M[pj, :] for all j.
            # M_all = M  # already computed
            # For j, M[p[j], :] is the row of M at index p[j].
            
            # Sum1_j = sum_k a[i,k] * (M[p[j], k] - row_pi[k]) for all k, minus corrections for k=i,j
            # Let's compute S1_all[j] = sum_k a[i,k] * M[p[j], k]  for all k
            # And S1_pi = sum_k a[i,k] * row_pi[k]  (scalar)
            # Then Sum1_j = S1_all[j] - S1_pi - corrections
            
            # S1_all[j] = sum_k a[i,k] * M[p[j], k] = (a[i, :] * M[p[j], :]).sum()
            # This is a[i] dot M[p[j]] for each j.
            # We can compute: S1_all = M @ a[i, :]. Wait, M is (n,n), M[loc,k] = b[loc,p[k]].
            # S1_all[j] = sum_k a[i,k] * M[p[j],k] = sum_k M[p[j],k] * a[i,k] = (M[p[j], :] @ a[i, :])
            # For all j: this is M[p, :] @ a[i, :]? No.
            # M[p[j], :] is the j-th row of M indexed by p[j]. So if we let N = M[p, :] (shape n,n where N[j,k] = M[p[j],k] = b[p[j],p[k]]),
            # then S1_all = N @ a[i, :]. Shape (n,).
            
            N = M[p, :]  # N[j,k] = b[p[j], p[k]], shape (n,n)
            # Note: N is actually the matrix B_p = b[p[:,None], p[None,:]] but transposed? 
            # N[j,k] = b[p[j], p[k]]. So N = b[np.ix_(p, p)]. Let's call it Bp.
            Bp = b[np.ix_(p, p)]  # Bp[j,k] = b[p[j], p[k]]
            
            # S1_all[j] = sum_k a[i,k] * Bp[j,k] = (Bp @ a[i, :])[j]? No, Bp[j,k] * a[i,k] summed over k = (Bp.T @ a[i,:])[j] if Bp is (n,n).
            # Actually: sum_k Bp[j,k] * a[i,k] = (Bp @ a[i, :])? No, that's sum_k Bp[j,k]*a[k,i] if a is not transposed.
            # Let's be careful: a[i,k] is row i of a. Bp[j,k] is row j of Bp.
            # sum_k Bp[j,k] * a[i,k] = dot(Bp[j,:], a[i,:])
            # For all j: this is Bp @ a[i, :].T? No.
            # S1_all = Bp @ a[i, :].T? Bp is (n,n), a[i,:] is (n,). Bp @ a[i,:].T gives (n,1). 
            # Actually, numpy: Bp @ a[i, :] does matrix-vector multiply: result[j] = sum_k Bp[j,k] * a[i,k]. Yes!
            S1_all = Bp @ a[i, :]  # (n,)
            S1_pi = (row_pi * a[i, :]).sum()  # scalar = sum_k b[pi,p[k]] * a[i,k]
            
            # Sum1_j = S1_all[j] - S1_pi, then subtract terms where k=i and k=j
            # Term k=i: a[i,i] * (Bp[j,i] - row_pi[i]) = a[i,i] * (b[pj,pi] - b[pi,pi])
            # Term k=j: a[i,j] * (Bp[j,j] - row_pi[j]) = a[i,j] * (b[pj,pj] - b[pi,pj])
            # So Sum1_j = (S1_all[j] - S1_pi) - a[i,i]*(b[pj,pi]-b[pi,pi]) - a[i,j]*(b[pj,pj]-b[pi,pj])
            
            # Similarly for Sum2_j = sum_k a[k,i] * (b[p[k],pj] - b[p[k],pi])
            # = sum_k a[k,i] * (Bp.T[k,j] - Bp.T[k,i])  since Bp.T[k,j] = b[p[j],p[k]]? No.
            # Bp[j,k] = b[p[j],p[k]], so Bp.T[k,j] = b[p[j],p[k]].
            # b[p[k],pj] = Bp[k,j] (since Bp[k,j] = b[p[k],p[j]] = b[p[k],pj]).
            # b[p[k],pi] = Bp[k,i].
            # So Sum2_j = sum_k a[k,i] * (Bp[k,j] - Bp[k,i])
            # = sum_k a[k,i] * Bp[k,j] - sum_k a[k,i] * Bp[k,i]
            # Let col_i_a = a[:, i]  # (n,)
            # Sum2_all[j] = sum_k a[k,i] * Bp[k,j] = (col_i_a @ Bp)[j] = (a[:,i] @ Bp)[j]
            # Sum2_pi = sum_k a[k,i] * Bp[k,i] = (a[:,i] @ Bp[:,i]).sum()
            
            col_i_a = a[:, i]  # (n,)
            Sum2_all = col_i_a @ Bp  # (n,)
            Sum2_pi = (col_i_a * Bp[:, i]).sum()
            
            # Sum2_j = (Sum2_all[j] - Sum2_pi) - corrections for k=i,j
            # k=i: a[i,i] * (Bp[i,j] - Bp[i,i]) = a[i,i] * (b[pi,pj] - b[pi,pi])
            # k=j: a[j,i] * (Bp[j,j] - Bp[j,i]) = a[j,i] * (b[pj,pj] - b[pj,pi])
            
            # Sum3_j = sum_k a[j,k] * (b[pi,p[k]] - b[pj,p[k]]) = sum_k a[j,k] * (row_pi[k] - Bp[j,k])
            # = (a[j,:] @ row_pi) - (a[j,:] @ Bp[j,:])
            # But this is for a specific j. For all j, we can't easily vectorize because a[j,:] varies.
            
            # Given the complexity, let me just do a straightforward O(n^3) pass with NumPy for inner loops.
            # n=256, n^3 = 16.7M. With NumPy, a full pass might take ~1-2 seconds. We can do ~50-100 passes.
            
            pass  # Skip the complex vectorization, use simpler approach

        # Simpler 2-opt: for each i, for each j>i, compute delta by direct formula using precomputed arrays.
        # Let me precompute:
        # Bp = b[np.ix_(p, p)]  # Bp[j,k] = b[p[j],p[k]]
        # Then delta_ij = 
        #   a[i,i]*(Bp[j,j]-Bp[i,i]) + a[i,j]*(Bp[j,i]-Bp[i,j]) + a[j,i]*(Bp[i,j]-Bp[j,i]) + a[j,j]*(Bp[i,i]-Bp[j,j])
        #   + sum_{k!=i,j} [a[i,k]*(Bp[j,k]-Bp[i,k]) + a[k,i]*(Bp[k,j]-Bp[k,i]) + a[j,k]*(Bp[i,k]-Bp[j,k]) + a[k,j]*(Bp[k,i]-Bp[k,j])]
        
        Bp = b[np.ix_(p, p)]  # (n,n), Bp[j,k] = b[p[j],p[k]]
        
        # For each i, compute delta for all j > i
        for i in range(n):
            if time.time() - t_start > time_budget:
                break
            best_delta = 0
            best_j = -1
            for j in range(i + 1, n):
                # Compute delta_ij
                # Pair terms
                delta = (a[i, i] * (Bp[j, j] - Bp[i, i]) +
                         a[i, j] * (Bp[j, i] - Bp[i, j]) +
                         a[j, i] * (Bp[i, j] - Bp[j, i]) +
                         a[j, j] * (Bp[i, i] - Bp[j, j]))
                
                # Sum over k != i, j
                # Vectorized over k:
                # Term 1: sum_k!=i,j a[i,k]*(Bp[j,k]-Bp[i,k])
                # Term 2: sum_k!=i,j a[k,i]*(Bp[k,j]-Bp[k,i])
                # Term 3: sum_k!=i,j a[j,k]*(Bp[i,k]-Bp[j,k])
                # Term 4: sum_k!=i,j a[k,j]*(Bp[k,i]-Bp[k,j])
                
                # Compute all k sums, then subtract k=i and k=j
                t1_all = (a[i, :] * (Bp[j, :] - Bp[i, :])).sum()
                t1_ii = a[i, i] * (Bp[j, i] - Bp[i, i])
                t1_ij = a[i, j] * (Bp[j, j] - Bp[i, j])
                t1 = t1_all - t1_ii - t1_ij
                
                t2_all = (a[:, i] * (Bp[:, j] - Bp[:, i])).sum()
                t2_ii = a[i, i] * (Bp[i, j] - Bp[i, i])
                t2_ij = a[j, i] * (Bp[j, j] - Bp[j, i])
                t2 = t2_all - t2_ii - t2_ij
                
                t3_all = (a[j, :] * (Bp[i, :] - Bp[j, :])).sum()
                t3_ii = a[j, i] * (Bp[i, i] - Bp[j, i])
                t3_ij = a[j, j] * (Bp[i, j] - Bp[j, j])
                t3 = t3_all - t3_ii - t3_ij
                
                t4_all = (a[:, j] * (Bp[:, i] - Bp[:, j])).sum()
                t4_ii = a[i, j] * (Bp[i, i] - Bp[i, j])
                t4_ij = a[j, j] * (Bp[j, i] - Bp[j, j])
                t4 = t4_all - t4_ii - t4_ij
                
                delta += t1 + t2 + t3 + t4
                
                if delta < best_delta:
                    best_delta = delta
                    best_j = j
            
            if best_j >= 0 and best_delta < 0:
                # Swap i and best_j
                p[i], p[best_j] = p[best_j], p[i]
                current_cost += best_delta
                # Update Bp
                Bp = b[np.ix_(p, p)]
                improved = True

    return p.tolist()
# EVOLVE-BLOCK-END
