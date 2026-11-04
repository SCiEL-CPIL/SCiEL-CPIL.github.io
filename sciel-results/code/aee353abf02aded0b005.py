"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _total_cost(a, b, p):
    """Compute the exact QAP cost for assignment p."""
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Efficient: (a @ b) evaluated at permutation
    # Let B = b[p[:, None], p[None, :]]  (permuted b)
    # cost = sum(a * B)
    B = b[np.ix_(p, p)]
    return int(np.sum(a * B))


def _compute_marginal_costs(a, b, p):
    """
    Compute the marginal cost matrix C where C[i, j] is the cost contribution
    if facility i is assigned to location j, given the current assignment p
    for all other facilities.
    
    The QAP cost is sum_{i,j} a[i,j] * b[p[i], p[j]].
    If we fix all assignments except facility i, the terms involving i are:
      sum_j a[i,j] * b[j_loc, p[j]]  +  sum_j a[j,i] * b[p[j], j_loc]
    where j_loc is the location assigned to facility i.
    
    So C[i, j_loc] = sum_k a[i,k] * b[j_loc, p[k]] + sum_k a[k,i] * b[p[k], j_loc]
    """
    n = len(p)
    # Precompute: for each facility i and location j, the marginal cost
    # C[i, j] = sum_k a[i,k] * b[j, p[k]] + sum_k a[k,i] * b[p[k], j]
    
    # First term: a @ b[:, p]  -> for each i, j: sum_k a[i,k] * b[j, p[k]]
    # b[:, p] is b with columns permuted by p
    b_col_perm = b[:, p]  # shape (n, n), column k is b[:, p[k]]
    # term1[i, j] = sum_k a[i,k] * b_col_perm[j, k] = (a @ b_col_perm.T)[i, j]
    term1 = a @ b_col_perm.T  # shape (n, n)
    
    # Second term: for each i, j: sum_k a[k,i] * b[p[k], j]
    # b[p, :] is b with rows permuted by p
    b_row_perm = b[p, :]  # shape (n, n), row k is b[p[k], :]
    # term2[i, j] = sum_k a[k,i] * b_row_perm[k, j] = (a.T @ b_row_perm)[i, j]
    term2 = a.T @ b_row_perm  # shape (n, n)
    
    C = term1 + term2
    return C


def _local_search(a, b, p, max_iter=500, time_limit=10.0):
    """
    2-opt style local search: try swapping two facilities' locations.
    Use vectorized evaluation of all swap pairs.
    """
    n = len(p)
    p = np.array(p, dtype=np.int64)
    best_cost = _total_cost(a, b, p)
    
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute marginal cost matrix
        C = _compute_marginal_costs(a, b, p)
        
        # For each pair (i, j), compute the change in cost if we swap p[i] and p[j]
        # Current cost contribution of i and j:
        #   C[i, p[i]] + C[j, p[j]] (but these include cross terms)
        # Actually, let's compute the delta more carefully.
        
        # The cost is sum_{u,v} a[u,v] * b[p[u], p[v]]
        # Swapping p[i] and p[j] changes terms where u or v is i or j.
        
        # Let's use a different approach: evaluate all swaps via the marginal matrix.
        # If we reassign facility i to location l (where l = p[j] for some j != i),
        # the change in cost is:
        #   new_contrib(i, l) - old_contrib(i, p[i])
        # where old_contrib(i, p[i]) = C[i, p[i]] (from marginal cost definition, 
        # but C includes the current assignment, so we need to be careful)
        
        # Actually, C[i, j] as defined above gives the contribution of facility i 
        # to the total cost if it's at location j, given others are fixed at p.
        # So the current total cost = sum_i C[i, p[i]] / 2? No, each pair (i,j) is counted twice.
        
        # Let me reconsider. The marginal cost C[i, j] = sum_k a[i,k] * b[j, p[k]] + sum_k a[k,i] * b[p[k], j]
        # The total cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # = sum_i sum_j a[i,j] * b[p[i], p[j]]
        
        # For a fixed assignment p, sum_i C[i, p[i]] = sum_i [sum_k a[i,k] * b[p[i], p[k]] + sum_k a[k,i] * b[p[k], p[i]]]
        # = sum_{i,k} a[i,k] * b[p[i], p[k]] + sum_{i,k} a[k,i] * b[p[k], p[i]]
        # = 2 * sum_{i,k} a[i,k] * b[p[i], p[k]]  (since the second sum is just a relabeling)
        # Wait, that's only if a is symmetric. In general:
        # sum_i sum_k a[i,k] * b[p[i], p[k]] + sum_i sum_k a[k,i] * b[p[k], p[i]]
        # Let me substitute in the second: sum_i sum_k a[k,i] * b[p[k], p[i]] = sum_k sum_i a[i,k] * b[p[i], p[k]] (swap i and k)
        # So yes, sum_i C[i, p[i]] = 2 * total_cost.
        
        # So total_cost = sum_i C[i, p[i]] / 2
        
        # If we reassign facility i to location j (where j = p[m] for some m), the new marginal for i at j is C[i, j].
        # But this C was computed with i at p[i], so it's not quite right for the new assignment.
        
        # For a proper 1-move: reassign i from p[i] to l. The change is:
        # delta = [sum_k a[i,k] * b[l, p[k]] + sum_k a[k,i] * b[p[k], l]] - [sum_k a[i,k] * b[p[i], p[k]] + sum_k a[k,i] * b[p[k], p[i]]]
        #        = C_new[i, l] - C_old[i, p[i]]
        # But C[i, l] as computed above already gives the first bracket (with i removed from its current position, but p still has i at p[i]).
        
        # This is getting complex. Let me just do a simpler approach: 
        # try all pairs of locations for each facility, but that's O(n^3).
        
        # Alternative: use the Hungarian algorithm on the marginal cost matrix to find a better permutation.
        # This is a greedy step: solve the linear assignment problem on C to get a new permutation.
        
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        
        new_cost = _total_cost(a, b, new_p)
        if new_cost < best_cost:
            p = new_p
            best_cost = new_cost
        else:
            # No improvement from the Hungarian step; try random swaps
            break
    
    return p, best_cost


def solve(a, b):
    n = len(a)
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
    
    best_p = p.copy()
    best_cost = _total_cost(a, b, p)
    
    # Local search with Hungarian-based reassignment
    p, cost = _local_search(a, b, p, max_iter=100, time_limit=60.0)
    if cost < best_cost:
        best_cost = cost
        best_p = p.copy()
    
    if time.time() > deadline:
        return best_p.tolist()
    
    # Try a few random restarts with local search
    rng = np.random.default_rng(42)
    for restart in range(5):
        if time.time() > deadline - 20.0:
            break
        
        # Random permutation
        p_rand = rng.permutation(n).astype(np.int64)
        p_ls, cost_ls = _local_search(a, b, p_rand, max_iter=50, time_limit=20.0)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
