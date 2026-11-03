"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p (facility i at location p[i])."""
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Vectorized: b_perm = b[np.ix_(p, p)]
    b_perm = b[np.ix_(p, p)]
    return int(np.sum(a * b_perm))


def _greedy_initial(a, b):
    """Rank-matching initial solution: heavy flow rows to low-distance rows."""
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _compute_grad(b, p):
    """
    Compute the gradient-like quantity for each facility i:
    grad[i] = sum_j a[i,j] * (sum over neighbors... )
    Actually, for QAP the change in cost when swapping p[i] and p[j] can be computed.
    But for a linearized surrogate, we can compute:
    For facility i placed at location L=p[i], the 'effective cost' contribution is:
      sum_j a[i,j] * b[L, p[j]]  +  sum_j a[j,i] * b[p[j], L]
    (since a[i,j] and a[j,i] both contribute with b[p[i],p[j]] and b[p[j],p[i]])
    
    A simpler surrogate: for each facility i, define
      g[i] = sum_j a[i,j] * b[p[i], p[j]]
    This is the row i of A times the permuted B row p[i].
    Then the total cost is roughly sum_i g[i] (with some double-counting).
    
    For 2-opt, we need the delta of swapping p[i] and p[j].
    """
    n = len(a) if 'a' in dir() else len(b)
    # We'll compute this inside the local search where a is available
    pass


def _local_search_2opt(a, b, p, time_budget=100.0, deadline=None):
    """
    2-opt local search for QAP. 
    Delta cost of swapping locations of facilities i and j (i.e., p[i] and p[j] swap):
    Let L_i = p[i], L_j = p[j].
    Old contribution involving i or j:
      sum_k a[i,k]*b[L_i, p[k]] + sum_k a[k,i]*b[p[k], L_i] + sum_k a[j,k]*b[L_j, p[k]] + sum_k a[k,j]*b[p[k], L_j]
      minus the double-counted a[i,j]*b[L_i,L_j] + a[j,i]*b[L_j,L_i] which were counted in both sums.
    Actually, let's define:
      C(i, L) = sum_k a[i,k]*b[L, p[k]] + sum_k a[k,i]*b[p[k], L] - a[i,i]*b[L,p[i]] ... 
    This is getting complex. Let me use a cleaner formulation.
    
    Total cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    
    When we swap p[i] and p[j], only terms involving i or j change.
    Let old_p = p.copy()
    Let L_i = old_p[i], L_j = old_p[j]
    
    Terms that change:
    For all k, the terms a[i,k]*b[L_i, p[k]] and a[k,i]*b[p[k], L_i] and a[j,k]*b[L_j, p[k]] and a[k,j]*b[p[k], L_j]
    But when k=i or k=j, we need to be careful.
    
    Let's define:
    S_i = sum_{k != i, k != j} a[i,k]*b[L_i, p[k]] + sum_{k != i, k != j} a[k,i]*b[p[k], L_i]
    S_j = sum_{k != i, k != j} a[j,k]*b[L_j, p[k]] + sum_{k != i, k != j} a[k,j]*b[p[k], L_j]
    S_ij = a[i,j]*b[L_i, L_j] + a[j,i]*b[L_j, L_i]
    
    After swap, p[i]=L_j, p[j]=L_i:
    S_i' = sum_{k != i, k != j} a[i,k]*b[L_j, p[k]] + sum_{k != i, k != j} a[k,i]*b[p[k], L_j]
    S_j' = sum_{k != i, k != j} a[j,k]*b[L_i, p[k]] + sum_{k != i, k != j} a[k,j]*b[p[k], L_i]
    S_ij' = a[i,j]*b[L_j, L_i] + a[j,i]*b[L_i, L_j]
    
    Delta = (S_i' + S_j' + S_ij') - (S_i + S_j + S_ij)
    
    We can precompute for each facility i and each location L:
      R[i, L] = sum_{k != i} a[i,k]*b[L, p[k]] + sum_{k != i} a[k,i]*b[p[k], L]
    But p changes as we go, so this is expensive to maintain.
    
    For simplicity and to stay within time, I'll do a simpler approach:
    Compute the delta on-the-fly for candidate pairs, but limit the number of candidates.
    """
    n = len(a)
    p = p.copy()
    best_cost = _qap_cost(a, b, p)
    
    # Precompute row sums for quick reference
    start_time = time.time()
    max_time = time_budget
    if deadline is not None:
        max_time = min(max_time, deadline - start_time)
    
    improved = True
    while improved:
        improved = False
        # Try all pairs (i, j) but in a random order to avoid bias
        # For n=256, there are ~32640 pairs. Computing delta for each is O(n) so total O(n^3) per pass.
        # 256^3 = 16.7M ops per pass. That's feasible in a few seconds.
        
        # Let's compute deltas for all pairs
        # For efficiency, precompute:
        # For each facility i, and for the current p, compute:
        #   out_i[L] = sum_k a[i,k] * b[L, p[k]]  for L in locations
        #   in_i[L] = sum_k a[k,i] * b[p[k], L]   for L in locations
        # Then R[i, L] = out_i[L] + in_i[L] - a[i,i]*b[L, p[i]] - ... 
        # Actually, when k=i, the term a[i,i]*b[L, p[i]] is included in out_i[L] if p[i]=L... 
        # This is getting complicated. Let me just do the direct computation for pairs.
        
        # Simpler: for each pair (i,j), compute delta directly.
        # To speed up, we can precompute for each i:
        #   A_i = a[i, :]  (row i of a)
        #   B_row[p[i]] = b[p[i], :]  (row p[i] of b)
        #   contrib_i = sum_k a[i,k] * b[p[i], p[k]]
        # But we need the cross terms.
        
        # Let me just do it the straightforward way for now, and optimize if needed.
        # For n=256, 32640 pairs * 256 operations = ~8.3M ops per pass. Should be < 1 second in numpy.
        
        # Precompute b rows for current p
        # b_perm[k, l] = b[p[k], p[l]]
        # But we need b[L, p[k]] for arbitrary L, so we need full b.
        
        # Let's compute deltas vectorized:
        # For pair (i, j):
        # old_cost_ij_terms = a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]]
        # For k not in {i,j}:
        #   old: a[i,k]*b[p[i],p[k]] + a[k,i]*b[p[k],p[i]] + a[j,k]*b[p[j],p[k]] + a[k,j]*b[p[k],p[j]]
        #   new: a[i,k]*b[p[j],p[k]] + a[k,i]*b[p[k],p[j]] + a[j,k]*b[p[i],p[k]] + a[k,j]*b[p[k],p[i]]
        
        # Let's precompute for each i:
        #   row_a_i = a[i, :]
        #   col_a_i = a[:, i]
        #   b_row_pi = b[p[i], :]  # row p[i] of b
        #   b_col_pi = b[:, p[i]]  # col p[i] of b
        
        # For a given i, define:
        #   F_i(L) = sum_k a[i,k]*b[L, p[k]] + sum_k a[k,i]*b[p[k], L]
        # Then for pair (i,j) with L_i=p[i], L_j=p[j]:
        #   delta = F_i(L_j) + F_j(L_i) + a[i,j]*b[L_j,L_i] + a[j,i]*b[L_i,L_j] 
        #           - F_i(L_i) - F_j(L_j) - a[i,j]*b[L_i,L_j] - a[j,i]*b[L_j,L_i]
        
        # Let's precompute F_i(L) for all i, L. This is n x n matrix.
        # F_i(L) = sum_k a[i,k]*b[L, p[k]] + sum_k a[k,i]*b[p[k], L]
        # Let Bp = b[np.ix_(range(n), p)]  # Bp[L, k] = b[L, p[k]]
        # Then sum_k a[i,k]*b[L, p[k]] = (a[i, :] @ Bp[L, :]) ... no, that's not right.
        # sum_k a[i,k] * b[L, p[k]] = dot(a[i, :], Bp[L, :])
        # So F_i(L) first term = dot(a[i, :], Bp[L, :])
        # Second term: sum_k a[k,i] * b[p[k], L] = dot(a[:, i], b[np.ix_(p, [L])[:, 0]])
        #   = dot(a[:, i], b[p, L])
        
        # Let's compute:
        # Bp = b[:, p]  # shape (n, n), Bp[L, k] = b[L, p[k]]
        # term1[i, L] = sum_k a[i,k] * Bp[L, k] = (a @ Bp.T)[i, L]  ... wait
        # sum_k a[i,k] * Bp[L, k] = (a[i, :] * Bp[L, :]).sum()
        # This is element-wise multiply and sum. We can do:
        # term1 = a @ Bp.T  # (n, n) where term1[i, L] = sum_k a[i,k] * Bp.T[k, L] = sum_k a[i,k] * Bp[L, k]
        # Yes!
        
        # term2[i, L] = sum_k a[k,i] * b[p[k], L]
        # b[p, L] is a vector of length n: [b[p[0], L], b[p[1], L], ...]
        # term2[i, L] = dot(a[:, i], b[p, L])
        # Let Bp_col = b[np.ix_(p, range(n))]  # Bp_col[k, L] = b[p[k], L]
        # term2 = a.T @ Bp_col  # (n, n) where term2[i, L] = sum_k a.T[i, k] * Bp_col[k, L] = sum_k a[k, i] * b[p[k], L]
        
        Bp = b[:, p]  # Bp[L, k] = b[L, p[k]]
        term1 = a @ Bp.T  # term1[i, L] = sum_k a[i,k] * b[L, p[k]]
        
        Bp_col = b[np.ix_(p, range(n))]  # Bp_col[k, L] = b[p[k], L]
        term2 = a.T @ Bp_col  # term2[i, L] = sum_k a[k,i] * b[p[k], L]
        
        F = term1 + term2  # F[i, L] = sum_k a[i,k]*b[L,p[k]] + sum_k a[k,i]*b[p[k],L]
        
        # Now for each pair (i, j):
        # L_i = p[i], L_j = p[j]
        # delta = F[i, L_j] + F[j, L_i] + a[i,j]*b[L_j,L_i] + a[j,i]*b[L_i,L_j] 
        #         - F[i, L_i] - F[j, L_j] - a[i,j]*b[L_i,L_j] - a[j,i]*b[L_j,L_i]
        
        # We can vectorize this over all pairs.
        # Let's create arrays for all pairs.
        # For efficiency, iterate over i and compute deltas for all j > i.
        
        for i in range(n - 1):
            L_i = p[i]
            Fi_Li = F[i, L_i]
            # For all j > i:
            js = np.arange(i + 1, n)
            L_js = p[js]
            # F[i, L_j] for all j
            Fi_Lj = F[i, L_js]
            # F[j, L_i] for all j
            Fj_Li = F[js, L_i]
            # a[i, j] for all j
            a_i_js = a[i, js]
            a_js_i = a[js, i]
            # b[L_j, L_i] for all j
            b_Lj_Li = b[L_js, L_i]
            # b[L_i, L_j] for all j
            b_Li_Lj = b[L_i, L_js]
            
            deltas = (Fi_Lj + Fj_Li + a_i_js * b_Lj_Li + a_js_i * b_Li_Lj) \
                     - (Fi_Li + F[js, L_js] + a_i_js * b_Li_Lj + a_js_i * b_Lj_Li)
            
            # Find the best (most negative) delta
            min_idx = np.argmin(deltas)
            if deltas[min_idx] < -1e-9:
                j = js[min_idx]
                # Swap p[i] and p[j]
                p[i], p[j] = p[j], p[i]
                best_cost += int(deltas[min_idx])
                improved = True
                # Recompute F for next iteration
                Bp = b[:, p]
                term1 = a @ Bp.T
                Bp_col = b[np.ix_(p, range(n))]
                term2 = a.T @ Bp_col
                F = term1 + term2
                # Check time
                if time.time() - start_time > max_time:
                    return p, best_cost
        
        # Check time
        if time.time() - start_time > max_time:
            break
    
    return p, best_cost


def _hungarian_reassign(a, b, p, deadline=None):
    """
    Use Hungarian algorithm on a linearized cost matrix to get a new assignment.
    Linearized cost: for facility i, the cost of placing it at location L is:
      c[i, L] = sum_k a[i,k] * b[L, p[k]] + sum_k a[k,i] * b[p[k], L]
    This is the same as F[i, L] computed above.
    Then solve the assignment problem on c.
    """
    n = len(a)
    Bp = b[:, p]
    term1 = a @ Bp.T
    Bp_col = b[np.ix_(p, range(n))]
    term2 = a.T @ Bp_col
    c = term1 + term2
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(c)
    new_p = np.empty(n, dtype=np.int64)
    new_p[row_ind] = col_ind
    
    new_cost = _qap_cost(a, b, new_p)
    old_cost = _qap_cost(a, b, p)
    
    if new_cost < old_cost:
        return new_p, new_cost
    return p, old_cost


def solve(a, b):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = len(a)
    
    # Initial solution
    p = _greedy_initial(a, b)
    best_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    
    # Local search
    p, cost = _local_search_2opt(a, b, p, time_budget=60.0, deadline=deadline)
    if cost < best_cost:
        best_cost = cost
        best_p = p.copy()
    
    # Check time
    if time.time() > deadline - 30.0:
        return best_p.tolist()
    
    # Hungarian reassignment + local search
    p, cost = _hungarian_reassign(a, b, p, deadline=deadline)
    if cost < best_cost:
        best_cost = cost
        best_p = p.copy()
    
    if time.time() > deadline - 20.0:
        return best_p.tolist()
    
    # Final polish
    p, cost = _local_search_2opt(a, b, p, time_budget=20.0, deadline=deadline)
    if cost < best_cost:
        best_cost = cost
        best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
