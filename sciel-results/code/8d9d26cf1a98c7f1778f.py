"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cost(a, b, p):
    """Compute exact QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    n = len(p)
    # b_perm[i, j] = b[p[i], p[j]]
    b_perm = b[np.ix_(p, p)]
    return int(np.sum(a * b_perm))


def _delta_2swap(a, b, p, i, j, row_sums_a, col_sums_a, b_row_sums, b_col_sums):
    """
    Compute cost change when swapping positions i and j in permutation p.
    p maps facility -> location. Swapping means:
      new_p[i] = old_p[j], new_p[j] = old_p[i], rest unchanged.
    
    We need: sum_{k} a[i,k]*(b[pi,pk] -> b[pj,pk]) + a[j,k]*(b[pj,pk] -> b[pi,pk])
              + a[k,i]*(b[pk,pi] -> b[pk,pj]) + a[k,j]*(b[pk,pj] -> b[pk,pi])
              - diagonal terms adjustments
    
    Let's define:
      pi = p[i], pj = p[j]
      
    The cost is C = sum_{u,v} a[u,v] * b[p[u], p[v]]
    
    When we swap p[i] and p[j]:
    - For u=i, v=j: was a[i,j]*b[pi,pj], now a[i,j]*b[pj,pi]
    - For u=j, v=i: was a[j,i]*b[pj,pi], now a[j,i]*b[pi,pj]
    - For u=i, v=k (k!=i,j): was a[i,k]*b[pi,pk], now a[i,k]*b[pj,pk]
    - For u=j, v=k (k!=i,j): was a[j,k]*b[pj,pk], now a[j,k]*b[pi,pk]
    - For u=k, v=i (k!=i,j): was a[k,i]*b[pk,pi], now a[k,i]*b[pk,pj]
    - For u=k, v=j (k!=i,j): was a[k,j]*b[pk,pj], now a[k,j]*b[pk,pi]
    
    Delta = sum over all affected terms of (new - old)
    
    Let's group:
    Term1 (i,j and j,i):
      a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
      = (a[i,j] - a[j,i]) * (b[pj,pi] - b[pi,pj])
    
    Term2 (i with others k!=i,j):
      sum_{k!=i,j} a[i,k] * (b[pj,pk] - b[pi,pk])
    
    Term3 (j with others k!=i,j):
      sum_{k!=i,j} a[j,k] * (b[pi,pk] - b[pj,pk])
      = - sum_{k!=i,j} a[j,k] * (b[pj,pk] - b[pi,pk])
    
    So Term2+Term3 = sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[pj,pk] - b[pi,pk])
    
    Term4 (others k!=i,j with i):
      sum_{k!=i,j} a[k,i] * (b[pk,pj] - b[pk,pi])
    
    Term5 (others k!=i,j with j):
      sum_{k!=i,j} a[k,j] * (b[pk,pi] - b[pk,pj])
      = - sum_{k!=i,j} a[k,j] * (b[pk,pj] - b[pk,pi])
    
    So Term4+Term5 = sum_{k!=i,j} (a[k,i] - a[k,j]) * (b[pk,pj] - b[pk,pi])
    
    Total delta = Term1 + Term2_3 + Term4_5
    
    Where:
      b[pj,pk] - b[pi,pk] = b_row[pj][pk] - b_row[pi][pk]
      b[pk,pj] - b[pk,pi] = b_col[pj][pk] - b_col[pi][pk]  (since b is not symmetric necessarily)
    
    Wait, b[pk,pj] - b[pk,pi] is the difference in column pj vs pi of row pk.
    Actually, let me re-index. b is the distance matrix. b[x,y] = distance from location x to y.
    
    For Term2: b[pj,pk] - b[pi,pk] is the difference between row pj and row pi of b, evaluated at column pk.
    For Term4: b[pk,pj] - b[pk,pi] is the difference between column pj and column pi of b, evaluated at row pk.
    
    We can precompute:
      b_row_diff[k] = b[pj,k] - b[pi,k] for all k  (this is b[pj,:] - b[pi,:])
      b_col_diff[k] = b[k,pj] - b[k,pi] for all k  (this is b[:,pj] - b[:,pi])
    
    Then:
      Term2_3 = sum_{k!=i,j} (a[i,k] - a[j,k]) * b_row_diff[k]
               = sum_{k} (a[i,k] - a[j,k]) * b_row_diff[k] - (a[i,i]-a[j,i])*b_row_diff[i] - (a[i,j]-a[j,j])*b_row_diff[j]
    
    But a[i,i] and a[j,j] are typically 0 in QAP. Let's assume they are 0.
    Also a[i,j] and a[j,i] are handled in Term1.
    
    Actually, let's just compute it directly with vectorized operations for speed.
    """
    pi = p[i]
    pj = p[j]
    
    # b_row_diff[k] = b[pj, k] - b[pi, k]
    b_row_diff = b[pj, :] - b[pi, :]
    # b_col_diff[k] = b[k, pj] - b[k, pi]
    b_col_diff = b[:, pj] - b[:, pi]
    
    # a_diff_row[k] = a[i, k] - a[j, k]
    a_diff_row = a[i, :] - a[j, :]
    # a_diff_col[k] = a[k, i] - a[k, j]
    a_diff_col = a[:, i] - a[:, j]
    
    # Term2_3 = sum_k a_diff_row[k] * b_row_diff[k]
    term23 = np.dot(a_diff_row, b_row_diff)
    
    # Term4_5 = sum_k a_diff_col[k] * b_col_diff[k]
    term45 = np.dot(a_diff_col, b_col_diff)
    
    # Term1 = (a[i,j] - a[j,i]) * (b[pj,pi] - b[pi,pj])
    term1 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Now we need to subtract the double-counted terms for k=i and k=j
    # In term23, when k=i: a_diff_row[i] * b_row_diff[i] = (a[i,i]-a[j,i]) * (b[pj,i]-b[pi,i])
    #   But this corresponds to the (i,i) and (j,i) terms which are partially in term1.
    #   Actually, the (i,i) term doesn't change since p[i] changes but... wait.
    
    # Let me reconsider. The full expansion should be:
    # delta = sum_{u,v} a[u,v] * (b[new_p[u], new_p[v]] - b[old_p[u], old_p[v]])
    # where new_p differs from old_p only at i and j.
    
    # Affected (u,v) pairs:
    # (i,j): a[i,j]*(b[pj,pi] - b[pi,pj])
    # (j,i): a[j,i]*(b[pi,pj] - b[pj,pi])
    # (i,k), k!=i,j: a[i,k]*(b[pj,pk] - b[pi,pk])
    # (j,k), k!=i,j: a[j,k]*(b[pi,pk] - b[pj,pk])
    # (k,i), k!=i,j: a[k,i]*(b[pk,pj] - b[pk,pi])
    # (k,j), k!=i,j: a[k,j]*(b[pk,pi] - b[pk,pj])
    
    # So delta = (a[i,j] - a[j,i])*(b[pj,pi] - b[pi,pj])
    #           + sum_{k!=i,j} (a[i,k] - a[j,k])*(b[pj,pk] - b[pi,pk])
    #           + sum_{k!=i,j} (a[k,i] - a[k,j])*(b[pk,pj] - b[pk,pi])
    
    # term23_full = sum_k (a[i,k]-a[j,k])*(b[pj,k]-b[pi,k])
    #   = sum_{k!=i,j} ... + (a[i,i]-a[j,i])*(b[pj,i]-b[pi,i]) + (a[i,j]-a[j,j])*(b[pj,j]-b[pi,j])
    
    # The k=i term in term23: (a[i,i]-a[j,i])*(b[pj,i]-b[pi,i])
    #   This is NOT part of the actual delta because when u=i,v=i, p[i] changes from pi to pj,
    #   so b[p[i],p[i]] changes from b[pi,pi] to b[pj,pj]. 
    #   Actually wait: for (u,v)=(i,i): old is b[pi,pi], new is b[pj,pj].
    #   So the (i,i) term contributes a[i,i]*(b[pj,pj]-b[pi,pi]).
    #   Similarly (j,j) contributes a[j,j]*(b[pi,pi]-b[pj,pj]).
    
    # I think the cleanest approach is to just compute the delta directly using the full sum
    # but restricted to affected indices. Let me use a different approach:
    
    # Compute the contribution of all pairs involving i or j before and after.
    
    # Before:
    #   pairs involving i: sum_v a[i,v]*b[pi,p[v]] + sum_u a[u,i]*b[p[u],pi] - a[i,i]*b[pi,pi]
    #   pairs involving j: sum_v a[j,v]*b[pj,p[v]] + sum_u a[u,j]*b[p[u],pj] - a[j,j]*b[pj,pj]
    #   pair (i,j) and (j,i) counted twice above, subtract once: a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
    
    # After swap:
    #   pairs involving i (now at pj): sum_v a[i,v]*b[pj,p[v]] + sum_u a[u,i]*b[p[u],pj] - a[i,i]*b[pj,pj]
    #   pairs involving j (now at pi): sum_v a[j,v]*b[pi,p[v]] + sum_u a[u,j]*b[p[u],pi] - a[j,j]*b[pi,pi]
    #   pair (i,j) and (j,i): a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
    
    # This is getting complex. Let me just use the direct formula with corrections.
    
    # The formula I had:
    # delta = term1 + sum_{k!=i,j}(a[i,k]-a[j,k])*(b[pj,k]-b[pi,k]) + sum_{k!=i,j}(a[k,i]-a[k,j])*(b[k,pj]-b[k,pi])
    
    # term23_full = np.dot(a_diff_row, b_row_diff) includes k=i and k=j
    # We need to subtract k=i and k=j from term23:
    #   k=i: (a[i,i]-a[j,i])*(b[pj,i]-b[pi,i])
    #   k=j: (a[i,j]-a[j,j])*(b[pj,j]-b[pi,j])
    
    # Similarly for term45:
    #   k=i: (a[i,i]-a[i,j])*(b[i,pj]-b[i,pi])
    #   k=j: (a[j,i]-a[j,j])*(b[j,pj]-b[j,pi])
    
    # And term1 already handles (i,j) and (j,i) correctly.
    # But the (i,i) and (j,j) diagonal terms:
    #   (i,i): a[i,i]*(b[pj,pj]-b[pi,pi])
    #   (j,j): a[j,j]*(b[pi,pi]-b[pj,pj])
    #   Net: (a[i,i]-a[j,j])*(b[pj,pj]-b[pi,pi])
    
    # So total delta = term1 
    #                  + (term23_full - (a[i,i]-a[j,i])*(b[pj,i]-b[pi,i]) - (a[i,j]-a[j,j])*(b[pj,j]-b[pi,j]))
    #                  + (term45_full - (a[i,i]-a[i,j])*(b[i,pj]-b[i,pi]) - (a[j,i]-a[j,j])*(b[j,pj]-b[j,pi]))
    #                  + (a[i,i]-a[j,j])*(b[pj,pj]-b[pi,pi])
    
    # Assuming a[i,i]=a[j,j]=0 (typical for QAP):
    # term23_correction = -(-a[j,i])*(b[pj,i]-b[pi,i]) - a[i,j]*(b[pj,j]-b[pi,j])
    #                    = a[j,i]*(b[pj,i]-b[pi,i]) - a[i,j]*(b[pj,j]-b[pi,j])
    # term45_correction = -(-a[i,j])*(b[i,pj]-b[i,pi]) - (-a[j,j])*(b[j,pj]-b[j,pi])  [a[j,j]=0]
    #                    = a[i,j]*(b[i,pj]-b[i,pi]) + 0
    # diag_correction = 0
    
    # Let me just verify with a small example or use the brute force for delta.
    
    # Actually, for reliability, let me compute the delta by evaluating the cost change
    # using precomputed row/column sums where possible, but for safety let's just do:
    
    # The most reliable vectorized approach:
    # delta = sum over all (u,v) of a[u,v]*(b[new_p[u],new_p[v]] - b[old_p[u],old_p[v]])
    # Only (u,v) where u in {i,j} or v in {i,j} are affected.
    
    # Let's compute:
    # old_contrib = sum_{v} a[i,v]*b[pi,p[v]] + sum_{u} a[u,i]*b[p[u],pi] + sum_{v} a[j,v]*b[pj,p[v]] + sum_{u} a[u,j]*b[p[u],pj] - a[i,i]*b[pi,pi] - a[j,j]*b[pj,pj] - a[i,j]*b[pi,pj] - a[j,i]*b[pj,pi]
    
    # new_contrib = sum_{v} a[i,v]*b[pj,p[v]] + sum_{u} a[u,i]*b[p[u],pj] + sum_{v} a[j,v]*b[pi,p[v]] + sum_{u} a[u,j]*b[p[u],pi] - a[i,i]*b[pj,pj] - a[j,j]*b[pi,pi] - a[i,j]*b[pj,pi] - a[j,i]*b[pi,pj]
    
    # delta = new_contrib - old_contrib
    
    # Let me compute each part:
    # sum_{v} a[i,v]*b[pi,p[v]] = sum_v a[i,v]*b[pi,p[v]]  -- this is a[i,:] dot b[pi, p]
    # sum_{u} a[u,i]*b[p[u],pi] = sum_u a[u,i]*b[p[u],pi]  -- this is a[:,i] dot b[p, pi]
    
    p_arr = p  # numpy array
    
    # For efficiency, precompute:
    # b_pi_p = b[pi, p]  -- row pi of b, indexed by p
    # b_p_pi = b[p, pi]  -- column pi of b, indexed by p
    # Similarly for pj.
    
    # old_i_row = a[i,:] @ b[pi, p_arr]
    # old_i_col = a[:,i] @ b[p_arr, pi]
    # old_j_row = a[j,:] @ b[pj, p_arr]
    # old_j_col = a[:,j] @ b[p_arr, pj]
    
    # new_i_row = a[i,:] @ b[pj, p_arr]
    # new_i_col = a[:,i] @ b[p_arr, pj]
    # new_j_row = a[j,:] @ b[pi, p_arr]
    # new_j_col = a[:,j] @ b[p_arr, pi]
    
    # This requires 8 dot products of length n. That's 8*n operations.
    # For n=150, that's 1200 operations per delta. With vectorization, should be fast.
    
    b_pi_p = b[pi, p_arr]
    b_p_pi = b[p_arr, pi]
    b_pj_p = b[pj, p_arr]
    b_p_pj = b[p_arr, pj]
    
    old_i = np.dot(a[i, :], b_pi_p) + np.dot(a[:, i], b_p_pi)
    new_i = np.dot(a[i, :], b_pj_p) + np.dot(a[:, i], b_p_pj)
    
    old_j = np.dot(a[j, :], b_pj_p) + np.dot(a[:, j], b_p_pj)
    new_j = np.dot(a[j, :], b_pi_p) + np.dot(a[:, j], b_p_pi)
    
    # Subtract double-counted diagonals and cross terms
    old_cross = a[i, j] * b[pi, pj] + a[j, i] * b[pj, pi]
    new_cross = a[i, j] * b[pj, pi] + a[j, i] * b[pi, pj]
    
    old_diag = a[i, i] * b[pi, pi] + a[j, j] * b[pj, pj]
    new_diag = a[i, i] * b[pj, pj] + a[j, j] * b[pi, pi]
    
    old_contrib = old_i + old_j - old_cross - old_diag
    new_contrib = new_i + new_j - new_cross - new_diag
    
    return new_contrib - old_contrib


def solve(a, b):
    n = len(a)
    deadline = time.time() + 110  # safety margin
    
    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    current_cost = _compute_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute some useful arrays
    a_sym = (a + a.T) / 2.0  # symmetrize for potential use, but we'll use original
    
    # 2-swap local search
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    break
                delta = _delta_2swap(a, b, p, i, j, None, None, None, None)
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
        if time.time() > deadline - 5:
            break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
