"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _compute_cost(a, b, p):
    """Compute QAP cost for permutation p: sum a[i,j] * b[p[i],p[j]]."""
    return np.sum(a * b[np.ix_(p, p)])


def _pairwise_swap_improve(a, b, p, max_iters=50000, time_limit=None, start_time=None):
    """Local search via pairwise swaps with delta evaluation."""
    n = len(p)
    p = np.array(p, dtype=np.int64)
    # Precompute row sums of a
    a_row_sums = a.sum(axis=1)
    
    if start_time is None:
        start_time = time.time()
    
    # Compute current cost
    bp = b[np.ix_(p, p)]
    cost = int(np.sum(a * bp))
    
    # For delta computation when swapping p[i] and p[j]:
    # delta = sum_k a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) + sum_k a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]])
    #        + sum_k a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]]) + sum_k a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]])
    #        + a[i,j]*(b[p[j],p[i]] - b[p[i],p[j]]) + a[j,i]*(b[p[i],p[j]] - b[p[j],p[i]])
    #        - a[i,j]*b[p[i],p[j]] - a[j,i]*b[p[j],p[i]]
    # Simplified: delta = sum_k [a[i,k] + a[k,i]] * (b[p[j],p[k]] - b[p[i],p[k]])
    #                  + sum_k [a[j,k] + a[k,j]] * (b[p[i],p[k]] - b[p[j],p[k]])
    # But we need to be careful with k=i and k=j terms.
    
    # Actually let's use a cleaner formulation:
    # Let u = p[i], v = p[j]. After swap, p[i]=v, p[j]=u.
    # The terms involving position i: sum_j a[i,j]*b[v, p[j]] (was sum_j a[i,j]*b[u,p[j]])
    # The terms involving position j: sum_j a[j,i]*b[p[j], v] (was sum_j a[j,i]*b[p[j],u])
    # Also cross terms a[i,j]*b[v,u] and a[j,i]*b[u,v] change.
    
    # Let's define:
    # row_a_i = a[i, :]  (flows from facility i)
    # col_a_i = a[:, i]  (flows to facility i)
    # row_a_j = a[j, :]
    # col_a_j = a[:, j]
    
    # Delta = sum_k a[i,k]*(b[v,p[k]] - b[u,p[k]]) + sum_k a[k,i]*(b[p[k],v] - b[p[k],u])
    #         + sum_k a[j,k]*(b[u,p[k]] - b[v,p[k]]) + sum_k a[k,j]*(b[p[k],u] - b[p[k],v])
    #         + a[i,j]*(b[v,u] - b[u,v]) + a[j,i]*(b[u,v] - b[v,u])
    #         - a[i,j]*b[u,v] - a[j,i]*b[v,u]  # wait, this is getting complicated
    
    # Let me just do it more carefully with the full expansion.
    # Original cost contribution involving i or j (as source or destination):
    # For each pair (x,y): a[x,y]*b[p[x],p[y]]
    # Terms where x=i: sum_y a[i,y]*b[u,p[y]]
    # Terms where y=i: sum_x a[x,i]*b[p[x],u]
    # Terms where x=j: sum_y a[j,y]*b[v,p[y]]
    # Terms where y=j: sum_x a[x,j]*b[p[x],v]
    # (The term (i,j) appears in both x=i sum and y=j sum, and (j,i) in x=j and y=i)
    
    # After swap (p[i]=v, p[j]=u):
    # Terms where x=i: sum_y a[i,y]*b[v,p'[y]] where p'[j]=u, p'[k]=p[k] for k!=i,j
    # = a[i,j]*b[v,u] + sum_{k!=i,j} a[i,k]*b[v,p[k]]
    # Terms where y=i: sum_x a[x,i]*b[p'[x],v] = a[j,i]*b[u,v] + sum_{k!=i,j} a[k,i]*b[p[k],v]
    # Terms where x=j: sum_y a[j,y]*b[u,p'[y]] = a[j,i]*b[u,v] + sum_{k!=i,j} a[j,k]*b[u,p[k]]
    # Terms where y=j: sum_x a[x,j]*b[p'[x],u] = a[i,j]*b[v,u] + sum_{k!=i,j} a[x,j]*b[p[k],u]
    
    # Hmm, this is getting messy. Let me just compute delta directly.
    
    # Actually, a simpler approach: precompute for each position i, the "row contribution" and "col contribution"
    # row_contrib[i] = sum_j a[i,j] * b[p[i], p[j]]
    # col_contrib[i] = sum_j a[j,i] * b[p[j], p[i]]
    # total = sum_i (row_contrib[i] + col_contrib[i]) / 2 ... no, that's not right either.
    
    # Let me just use a different approach: compute the delta by directly evaluating.
    
    # For efficiency, precompute:
    # For each position i, define:
    #   R[i] = sum_j a[i,j] * b[p[i], p[j]]  (outflow from facility i to all locations)
    #   C[i] = sum_j a[j,i] * b[p[j], p[i]]  (inflow to facility i from all locations)
    # Total cost = sum_i R[i] = sum_i C[i]
    
    # When we swap positions i and j (u=p[i], v=p[j]):
    # New R[i] = a[i,j]*b[v,u] + sum_{k!=i,j} a[i,k]*b[v,p[k]]
    # Old R[i] = a[i,j]*b[u,v] + sum_{k!=i,j} a[i,k]*b[u,p[k]]
    # Delta_R_i = sum_{k!=i,j} a[i,k]*(b[v,p[k]] - b[u,p[k]]) + a[i,j]*(b[v,u] - b[u,v])
    
    # Similarly for other terms. This is O(n) per swap evaluation.
    
    # Let's just do a simpler but still efficient approach:
    # Precompute b_p = b[np.ix_(p,p)]
    # For swap of positions i,j:
    #   u = p[i], v = p[j]
    #   delta = 0
    #   for k in range(n):
    #       if k == i or k == j: continue
    #       delta += a[i,k]*(b[v,p[k]] - b[u,p[k]])
    #       delta += a[k,i]*(b[p[k],v] - b[p[k],u])
    #       delta += a[j,k]*(b[u,p[k]] - b[v,p[k]])
    #       delta += a[k,j]*(b[p[k],u] - b[p[k],v])
    #   delta += a[i,j]*(b[v,u] - b[u,v])
    #   delta += a[j,i]*(b[u,v] - b[v,u])
    #   delta -= a[i,j]*b[u,v] + a[j,i]*b[v,u]  # wait no...
    
    # I think I'm overcomplicating this. Let me just use a vectorized delta computation.
    
    n = len(p)
    p = p.copy()
    bp = b[np.ix_(p, p)]
    cost = int(np.sum(a * bp))
    
    # Precompute a_sym = a + a.T for convenience? No, let's keep it general.
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time_limit is not None and (time.time() - start_time) > time_limit - 2.0:
            break
        
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Try all pairs (i, j) with i < j
        for i in range(n):
            for j in range(i + 1, n):
                u = p[i]
                v = p[j]
                if u == v:
                    continue
                
                # Compute delta for swapping positions i and j
                # Terms involving row i (source i):
                # Old: sum_k a[i,k]*b[u,p[k]]
                # New: sum_k a[i,k]*b[v,p[k]] (with p[j]=u instead of v)
                # So new = a[i,j]*b[v,u] + sum_{k!=i,j} a[i,k]*b[v,p[k]]
                # Old = a[i,j]*b[u,v] + sum_{k!=i,j} a[i,k]*b[u,p[k]]
                
                # Vectorized:
                # delta_row_i = a[i,:] @ (b[v,:] - b[u,:]) but need to handle p correctly
                # Actually: sum_k a[i,k]*b[v,p[k]] - sum_k a[i,k]*b[u,p[k]]
                # = sum_k a[i,k]*(b[v,p[k]] - b[u,p[k]])
                # = a[i,:] @ (b[v,p] - b[u,p])
                # where b[v,p] means b[v, p[0]], b[v, p[1]], ...
                
                delta_row_i = np.dot(a[i, :], b[v, p] - b[u, p])
                delta_col_i = np.dot(a[:, i], b[p, v] - b[p, u])
                delta_row_j = np.dot(a[j, :], b[u, p] - b[v, p])
                delta_col_j = np.dot(a[:, j], b[p, u] - b[p, v])
                
                # But this double-counts the (i,j) and (j,i) terms, and also the diagonal-ish terms.
                # Let's verify: 
                # delta_row_i includes k=j: a[i,j]*(b[v,p[j]] - b[u,p[j]]) = a[i,j]*(b[v,v] - b[u,v]) -- WRONG, p[j] is v, not p[j] after swap
                # 
                # The issue is that b[v,p] uses the OLD p, but after swap p[j] changes.
                # So we need to be more careful.
                
                # Let me redo this properly.
                # After swap: p_new[i] = v, p_new[j] = u, p_new[k] = p[k] for k != i,j
                # 
                # Cost = sum_{x,y} a[x,y] * b[p_new[x], p_new[y]]
                # 
                # The only terms that change are those where x or y is i or j.
                # 
                # Terms with x=i: sum_y a[i,y] * b[v, p_new[y]]
                #   = a[i,i]*b[v,v] + a[i,j]*b[v,u] + sum_{k!=i,j} a[i,k]*b[v,p[k]]
                # Terms with y=i: sum_x a[x,i] * b[p_new[x], v]
                #   = a[i,i]*b[v,v] + a[j,i]*b[u,v] + sum_{k!=i,j} a[k,i]*b[p[k],v]
                # Terms with x=j: sum_y a[j,y] * b[u, p_new[y]]
                #   = a[j,i]*b[u,v] + a[j,j]*b[u,u] + sum_{k!=i,j} a[j,k]*b[u,p[k]]
                # Terms with y=j: sum_x a[x,j] * b[p_new[x], u]
                #   = a[i,j]*b[v,u] + a[j,j]*b[u,u] + sum_{k!=i,j} a[x,j]*b[p[k],u]
                #
                # Old corresponding terms:
                # Terms with x=i: sum_y a[i,y] * b[u, p[y]]
                #   = a[i,i]*b[u,u] + a[i,j]*b[u,v] + sum_{k!=i,j} a[i,k]*b[u,p[k]]
                # Terms with y=i: sum_x a[x,i] * b[p[x], u]
                #   = a[i,i]*b[u,u] + a[j,i]*b[p[j],u] + sum_{k!=i,j} a[k,i]*b[p[k],u]
                #   = a[i,i]*b[u,u] + a[j,i]*b[v,u] + sum_{k!=i,j} a[k,i]*b[p[k],u]
                # Terms with x=j: sum_y a[j,y] * b[v, p[y]]
                #   = a[j,i]*b[v,u] + a[j,j]*b[v,v] + sum_{k!=i,j} a[j,k]*b[v,p[k]]
                # Terms with y=j: sum_x a[x,j] * b[p[x], v]
                #   = a[i,j]*b[u,v] + a[j,j]*b[v,v] + sum_{k!=i,j} a[x,j]*b[p[k],v]
                #
                # Delta = (new - old) for all these terms, but we need to be careful not to double count.
                # Actually, the total cost is sum_{x,y} a[x,y]*b[p[x],p[y]], and each (x,y) pair is counted once.
                # The terms that change are exactly those where x in {i,j} or y in {i,j}.
                # But (i,j) appears in both x=i and y=j, etc.
                # 
                # Let's just compute the total delta as:
                # delta = [sum_y a[i,y]*b[v,p_new[y]] + sum_x a[x,i]*b[p_new[x],v] + sum_y a[j,y]*b[u,p_new[y]] + sum_x a[x,j]*b[p_new[x],u]]
                #       - [sum_y a[i,y]*b[u,p[y]] + sum_x a[x,i]*b[p[x],u] + sum_y a[j,y]*b[v,p[y]] + sum_x a[x,j]*b[p[x],v]]
                #       - [a[i,j]*b[v,u] + a[j,i]*b[u,v] + a[i,j]*b[v,u] + a[j,i]*b[u,v]]  # subtract double-counted
                #       + [a[i,j]*b[u,v] + a[j,i]*b[v,u] + a[i,j]*b[u,v] + a[j,i]*b[v,u]]  # add back old double-counted
                #
                # This is getting too complicated analytically. Let me just do it numerically with precomputed vectors.
                
                # Simpler: just compute the difference directly using precomputed arrays.
                # Precompute for each location l: b[l, p] and b[p, l]
                # b_l_p = b[l, p] for all l -- this is just b[:, p] which is b with columns permuted
                # b_p_l = b[p, l] for all l -- this is b[p, :] which is b with rows permuted
                
                # Let me precompute these once outside the loop.
                break  # Break out of inner loop to restructure
            break  # Break out of outer loop
        break  # Break out of while loop
    
    # Let me rewrite this more carefully with precomputation.
    p = p.copy()
    bp = b[np.ix_(p, p)]
    cost = int(np.sum(a * bp))
    
    # Precompute: for each location l, the vector b[l, p] (row l of b, columns permuted by p)
    # and b[p, l] (columns of b, rows permuted by p)
    # b_l_p[l, k] = b[l, p[k]]
    # b_p_l[l, k] = b[p[l], k]
    # Actually: b[:, p] gives matrix where (l,k) entry is b[l, p[k]]
    # b[p, :] gives matrix where (l,k) entry is b[p[l], k]
    
    b_cols_p = b[:, p]  # shape (n,n), entry (l,k) = b[l, p[k]]
    b_rows_p = b[p, :]  # shape (n,n), entry (l,k) = b[p[l], k]
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time_limit is not None and (time.time() - start_time) > time_limit - 2.0:
            break
        
        best_delta = 0
        best_i, best_j = -1, -1
        
        for i in range(n):
            for j in range(i + 1, n):
                u = p[i]
                v = p[j]
                
                # Delta computation:
                # When swapping p[i] and p[j] (u <-> v):
                # 
                # For each k != i,j:
                #   a[i,k]: b[v,p[k]] - b[u,p[k]] = b_cols_p[v,k] - b_cols_p[u,k]
                #   a[k,i]: b[p[k],v] - b[p[k],u] = b_rows_p[k,v] - b_rows_p[k,u]
                #   a[j,k]: b[u,p[k]] - b[v,p[k]] = b_cols_p[u,k] - b_cols_p[v,k]
                #   a[k,j]: b[p[k],u] - b[p[k],v] = b_rows_p[k,u] - b_rows_p[k,v]
                # 
                # For k=i and k=j, we handle separately.
                # 
                # Sum over k != i,j of:
                #   (a[i,k] - a[j,k]) * (b_cols_p[v,k] - b_cols_p[u,k])
                #   + (a[k,i] - a[k,j]) * (b_rows_p[k,v] - b_rows_p[k,u])
                #
                # Plus boundary terms for k=i and k=j:
                # k=i in the first sum: (a[i,i] - a[j,i]) * (b_cols_p[v,i] - b_cols_p[u,i])
                #   but b_cols_p[v,i] = b[v, p[i]] = b[v, u], b_cols_p[u,i] = b[u, p[i]] = b[u, u]
                #   Wait, p[i] = u, so b_cols_p[v,i] = b[v, p[i]] = b[v,u], b_cols_p[u,i] = b[u, p[i]] = b[u,u]
                #   Hmm, this doesn't seem right for the swap.
                
                # I think the cleanest way is to just compute the delta by evaluating the changed terms.
                # Let me use a different approach: compute the cost change directly.
                
                # The terms that change are those with x in {i,j} or y in {i,j}.
                # Let's compute:
                # old_terms = sum_{x in {i,j} or y in {i,j}} a[x,y]*b[p[x],p[y]]  (but avoid double counting)
                # new_terms = sum_{x in {i,j} or y in {i,j}} a[x,y]*b[p_new[x],p_new[y]]
                
                # Actually, let's just compute:
                # delta = sum_{y} a[i,y]*(b[v,p_new[y]] - b[u,p[y]])
                #       + sum_{x} a[x,i]*(b[p_new[x],v] - b[p[x],u])
                #       + sum_{y} a[j,y]*(b[u,p_new[y]] - b[v,p[y]])
                #       + sum_{x} a[x,j]*(b[p_new[x],u] - b[p[x],v])
                #       - (a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u])  # correction for double counting in the sums above
                #         + a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u]))
                # 
                # This is still messy. Let me just do it brute force for the delta since n=150.
                
                # Brute force delta:
                # Create p_new = p with i,j swapped
                # Compute sum_{x,y: x in {i,j} or y in {i,j}} a[x,y]*(b[p_new[x],p_new[y]] - b[p[x],p[y]])
                # But this is O(n) which is fine.
                
                p_new = p.copy()
                p_new[i], p_new[j] = v, u
                
                # Compute delta for all (x,y) where x in {i,j} or y in {i,j}
                # To avoid double counting, let's just compute the full sum difference for these terms.
                # Terms with x=i: sum_y a[i,y]*b[p_new[i], p_new[y]] - sum_y a[i,y]*b[p[i], p[y]]
                # Terms with x=j: sum_y a[j,y]*b[p_new[j], p_new[y]] - sum_y a[j,y]*b[p[j], p[y]]
                # Terms with y=i (x != i,j): sum_{x!=i,j} a[x,i]*b[p_new[x], p_new[i]] - sum_{x!=i,j} a[x,i]*b[p[x], p[i]]
                # Terms with y=j (x != i,j): sum_{x!=i,j} a[x,j]*b[p_new[x], p_new[j]] - sum_{x!=i,j} a[x,j]*b[p[x], p[j]]
                
                # Note: (i,i), (i,j), (j,i), (j,j) are covered in the first two sums.
                # (x,i) for x!=i,j and (x,j) for x!=i,j are in the last two sums.
                
                delta = 0
                # x=i terms
                delta += np.dot(a[i, :], b[v, p_new] - b[u, p])
                # x=j terms
                delta += np.dot(a[j, :], b[u, p_new] - b[v, p])
                # y=i terms (x != i, j)
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                delta += np.dot(a[mask, i], b[p_new[mask], v] - b[p[mask], u])
                # y=j terms (x != i, j)
                delta += np.dot(a[mask, j], b[p_new[mask], u] - b[p[mask], v])
                
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_i, best_j = i, j
        
        if best_i >= 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            cost += int(best_delta)
            # Update bp
            u, v = p[best_i], p[best_j]
            # Update b_cols_p and b_rows_p
            b_cols_p[u, best_i] = b[u, v]
            b_cols_p[v, best_j] = b[v, u]
            b_cols_p[u, best_j] = b[u, u]
            b_cols_p[v, best_i] = b[v, v]
            # Actually this is getting complicated. Let me just recompute bp when needed.
            bp = b[np.ix_(p, p)]
            improved = True
    
    return p.tolist(), cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Phase 1: Initial solution via rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Phase 2: Hungarian method refinement on a reduced problem
    # Compute the cost matrix for the assignment problem
    # c[i,j] = sum_k a[i,k] * b[j, p[k]] + sum_k a[k,i] * b[p[k], j]
    # This is the "reduced cost" for assigning facility i to location j given current p for others
    # Actually, for a proper Hungarian refinement, we'd need to solve the full assignment,
    # but that's O(n^3) = 3.375M which is fast.
    
    # Let's do a few rounds of Hungarian-based improvement
    for round_num in range(3):
        if time.time() > deadline - 30.0:
            break
        
        # Compute reduced cost matrix
        # For facility i assigned to location j, the contribution is:
        # sum_k a[i,k]*b[j,p[k]] + sum_k a[k,i]*b[p[k],j] - a[i,i]*b[j,j] (to avoid double count)
        # Actually, the total cost is sum_{i,j} a[i,j]*b[p[i],p[j]]
        # If we fix all assignments except facility i->j, the marginal cost is:
        # sum_k a[i,k]*b[j,p[k]] + sum_k a[k,i]*b[p[k],j] - a[i,i]*b[j,j]
        # But this isn't quite right for a global assignment.
        
        # Let's use a different approach: compute the full cost matrix for the linear relaxation
        # c[i,j] = sum_k a[i,k]*b[j,k] + sum_k a[k,i]*b[k,j]
        # This is an upper bound on the marginal cost and can be used for Hungarian.
        
        # Actually, for QAP, the standard approach is to use the "reduced cost" from the
        # current solution. Let's just do pairwise swaps which is more reliable.
        
        # Do pairwise swap local search
        p_list, cost = _pairwise_swap_improve(a, b, p, max_iters=20000, time_limit=deadline - start_time, start_time=start_time)
        p = np.array(p_list, dtype=np.int64)
    
    # Phase 3: Simulated annealing for further improvement
    if time.time() < deadline - 10.0:
        p = np.array(p, dtype=np.int64)
        bp = b[np.ix_(p, p)]
        cost = int(np.sum(a * bp))
        
        T = max(1.0, cost * 1e-6)
        T_min = 0.5
        alpha = 0.9995
        n_iter = 0
        max_sa_iter = 500000
        
        while T > T_min and n_iter < max_sa_iter:
            if time.time() > deadline - 5.0:
                break
            
            # Random swap
            i, j = np.random.randint(0, n, 2)
            if i == j:
                continue
            
            u, v = p[i], p[j]
            
            # Compute delta
            p_new = p.copy()
            p_new[i], p_new[j] = v, u
            
            delta = 0
            delta += np.dot(a[i, :], b[v, p_new] - b[u, p])
            delta += np.dot(a[j, :], b[u, p_new] - b[v, p])
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            delta += np.dot(a[mask, i], b[p_new[mask], v] - b[p[mask], u])
            delta += np.dot(a[mask, j], b[p_new[mask], u] - b[p[mask], v])
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                p = p_new
                cost += int(delta)
            
            T *= alpha
            n_iter += 1
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 2.0:
        p_list, cost = _pairwise_swap_improve(a, b, p, max_iters=10000, time_limit=deadline - start_time, start_time=start_time)
        p = np.array(p_list, dtype=np.int64)
    
    return p.tolist()
# EVOLVE-BLOCK-END
