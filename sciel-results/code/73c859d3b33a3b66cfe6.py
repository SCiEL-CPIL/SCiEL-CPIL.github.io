"""QAP tai256c solver: iterated 2-opt local search with fast cost evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 110.0  # safety margin under 120s

    # Precompute the cost matrix C[i,j] = sum_k a[i,k] * b[j, p[k]]
    # Actually, cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # For a 2-opt swap of p[u] and p[v], the delta can be computed efficiently.
    #
    # Let's use a different approach: maintain the assignment and compute cost deltas.
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    #
    # If we swap positions u and v (i.e., facilities u and v swap locations),
    # the change in cost involves terms where i or j is u or v.
    #
    # Actually, p[i] = location of facility i. So swapping p[u] and p[v] means
    # facility u goes to location p[v] and facility v goes to location p[u].
    #
    # Let's define: for each facility i, its "row contribution" to cost is
    # sum_j a[i,j] * b[p[i], p[j]]
    # and "column contribution" is sum_i a[i,j] * b[p[i], p[j]]
    # But these overlap at (i,j) term.
    #
    # Total cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
    #
    # For a swap of p[u] <-> p[v], the affected terms are those where i=u or j=u or i=v or j=v.
    # Let's compute delta exactly.
    #
    # Before swap:
    #   Terms with i=u: sum_j a[u,j] * b[p[u], p[j]]  (j != u,v)
    #   Terms with j=u: sum_i a[i,u] * b[p[i], p[u]]  (i != u,v)
    #   Terms with i=v: sum_j a[v,j] * b[p[v], p[j]]  (j != u,v)
    #   Terms with j=v: sum_i a[i,v] * b[p[i], p[v]]  (i != u,v)
    #   Terms (u,v): a[u,v] * b[p[u], p[v]]
    #   Terms (v,u): a[v,u] * b[p[v], p[u]]
    #   Terms (u,u): a[u,u] * b[p[u], p[u]]
    #   Terms (v,v): a[v,v] * b[p[v], p[v]]
    #
    # After swap (p[u] and p[v] exchanged):
    #   Terms with i=u: sum_j a[u,j] * b[p[v], p[j]]  (j != u,v)
    #   Terms with j=u: sum_i a[i,u] * b[p[i], p[v]]  (i != u,v)
    #   Terms with i=v: sum_j a[v,j] * b[p[u], p[j]]  (j != u,v)
    #   Terms with j=v: sum_i a[i,v] * b[p[i], p[u]]  (i != u,v)
    #   Terms (u,v): a[u,v] * b[p[v], p[u]]
    #   Terms (v,u): a[v,u] * b[p[u], p[v]]
    #   Terms (u,u): a[u,u] * b[p[v], p[v]]
    #   Terms (v,v): a[v,v] * b[p[u], p[u]]
    #
    # This is getting complex. Let me use a simpler O(n) delta computation.
    #
    # Actually, let me just compute the full cost incrementally.
    # For n=256, full cost is O(n^2) = 65536 operations.
    # A 2-opt evaluation with delta is O(n).
    #
    # Let me precompute for each facility i:
    #   row_cost[i] = sum_j a[i,j] * b[p[i], p[j]]
    # Then total cost = sum_i row_cost[i]
    #
    # When we swap p[u] and p[v]:
    # For facility u: row_cost[u] changes from sum_j a[u,j]*b[old_pu, p[j]] to sum_j a[u,j]*b[old_pv, p[j]]
    #   but p[j] for j=u is now old_pv, and for j=v is now old_pu.
    #
    # This is still complex. Let me just use a direct O(n^2) cost function
    # and do a limited number of 2-opt passes. With n=256, n^2=65536, 
    # and if each 2-opt pass takes ~n^2 * constant, we can do many passes.

    def compute_cost(p):
        # p[i] = location of facility i
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Use numpy: create matrix B_p where B_p[i,j] = b[p[i], p[j]]
        # Then cost = sum(a * B_p)
        # B_p = b[np.ix_(p, p)]
        return np.sum(a * b[np.ix_(p, p)])

    # Better initial solution: use a greedy construction
    # Start with rank-matching, then improve with local search
    
    # Initial: rank matching based on row sums of a and column sums of b
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)  # sum over columns for each row of b
    # Actually b[p[i], p[j]] - we want heavy flow pairs to have small distance
    # Sort facilities by total flow (row sum of a) descending
    # Sort locations by total distance (row sum of b) ascending
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
    
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    
    # 2-opt local search
    # For each pair (i, j), try swapping p[i] and p[j]
    # Compute delta in O(n)
    
    def two_opt_pass(p, cost, max_iter=None):
        """One full pass of 2-opt. Returns improved p and cost."""
        n = len(p)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return p, cost
                    # Try swapping p[i] and p[j]
                    p[i], p[j] = p[j], p[i]
                    new_cost = compute_cost(p)
                    if new_cost < cost:
                        cost = new_cost
                        improved = True
                    else:
                        p[i], p[j] = p[j], p[i]
                    if max_iter is not None and improved:
                        max_iter -= 1
                        if max_iter <= 0:
                            return p, cost
        return p, cost

    # Do a few passes of 2-opt
    p, cost = two_opt_pass(p.copy(), best_cost)
    if cost < best_cost:
        best_p = p.copy()
        best_cost = cost

    # Iterated local search: perturb and re-optimize
    while time.time() < deadline:
        # Perturb: random swap of 2-5 pairs
        p = best_p.copy()
        n_swaps = np.random.randint(2, 6)
        for _ in range(n_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
        
        # Local search from perturbed solution
        p, cost = two_opt_pass(p, compute_cost(p))
        
        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
