"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    # p is a list/array of length n, p[i] = location of facility i
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Vectorized: create matrix Bp where Bp[i,j] = b[p[i], p[j]]
    # Then cost = sum(a * Bp)
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))


def _delta_swap(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in the assignment p.
    p is a list/array; p[i] and p[j] are the locations.
    
    The cost contribution involving facility i is:
      sum_k a[i,k] * b[p[i], p[k]] + sum_k a[k,i] * b[p[k], p[i]]
    Similarly for j.
    
    After swapping, facility i goes to p[j] and facility j goes to p[i].
    
    Delta = new_cost - old_cost for the terms involving i and j.
    """
    pi = p[i]
    pj = p[j]
    
    # Terms involving i (as row) and j (as row), and i (as col) and j (as col)
    # Let's compute the change carefully.
    
    # Old contribution of pair (i, j) and (j, i) in the sum:
    # sum_{i,j} a[i,j]*b[p[i],p[j]] includes:
    #   a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi] + a[i,i]*b[pi,pi] + a[j,j]*b[pj,pj]
    #   plus terms where i or j is one index and the other is some k != i,j
    
    # New contribution after swap:
    #   a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj] + a[i,i]*b[pj,pj] + a[j,j]*b[pi,pi]
    #   plus terms where i or j is one index and the other is some k != i,j
    
    # For k != i, j:
    # Old: a[i,k]*b[pi,p[k]] + a[k,i]*b[p[k],pi] + a[j,k]*b[pj,p[k]] + a[k,j]*b[p[k],pj]
    # New: a[i,k]*b[pj,p[k]] + a[k,i]*b[p[k],pj] + a[j,k]*b[pi,p[k]] + a[k,j]*b[p[k],pi]
    
    # Delta for k != i,j:
    # sum_{k != i,j} [a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + a[k,i]*(b[p[k],pj] - b[p[k],pi])
    #                + a[j,k]*(b[pi,p[k]] - b[pj,p[k]]) + a[k,j]*(b[p[k],pi] - b[p[k],pj])]
    
    # This is: sum_{k != i,j} [ (a[i,k] - a[j,k])*(b[pj,p[k]] - b[pi,p[k]])
    #                         + (a[k,i] - a[k,j])*(b[p[k],pj] - b[p[k],pi]) ]
    
    # And for the (i,j) and (j,i) terms plus self terms:
    # Delta_self = a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
    #             + a[i,i]*(b[pj,pj] - b[pi,pi]) + a[j,j]*(b[pi,pi] - b[pj,pj])
    
    # Let's vectorize the k != i,j part.
    # Create a mask excluding i and j.
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    
    # b[pj, p[mask]] - b[pi, p[mask]]
    diff1 = b[pj, p[mask]] - b[pi, p[mask]]
    # b[p[mask], pj] - b[p[mask], pi]
    diff2 = b[p[mask], pj] - b[p[mask], pi]
    
    a_diff_row = a[i, mask] - a[j, mask]
    a_diff_col = a[mask, i] - a[mask, j]
    
    delta_k = np.sum(a_diff_row * diff1) + np.sum(a_diff_col * diff2)
    
    # Self terms
    delta_self = (a[i, j] * (b[pj, pi] - b[pi, pj])
                  + a[j, i] * (b[pi, pj] - b[pj, pi])
                  + a[i, i] * (b[pj, pj] - b[pi, pi])
                  + a[j, j] * (b[pi, pi] - b[pj, pj]))
    
    return int(delta_k + delta_self)


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin
    
    # Initial assignment: rank-matching heuristic
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    best_cost = _cost(a, b, p)
    best_p = p.copy()
    
    # Local search with 2-opt swaps
    # Use a list to track visited pairs to avoid cycles
    # We'll do multiple passes until no improvement or deadline
    
    # Precompute some things for speed? For n=100, full O(n^2) delta eval per pass is 10000 evals.
    # Each delta eval is O(n) vectorized, so ~10000*100 = 1e6 ops per pass. That's fast.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all pairs
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    # Accept the swap
                    p[i], p[j] = p[j], p[i]
                    best_cost += delta
                    improved = True
                    # Don't break; continue searching
        if time.time() > deadline:
            break
    
    # If we still have time, try a different initial solution and local search
    if time.time() < deadline - 20:
        # Random restarts
        rng = np.random.default_rng(42)
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            p_r = rng.permutation(n)
            cost_r = _cost(a, b, p_r)
            # Local search from random start
            improved_r = True
            while improved_r and time.time() < deadline - 5:
                improved_r = False
                for i in range(n):
                    if time.time() > deadline - 5:
                        break
                    for j in range(i + 1, n):
                        delta = _delta_swap(a, b, p_r, i, j)
                        if delta < 0:
                            p_r[i], p_r[j] = p_r[j], p_r[i]
                            cost_r += delta
                            improved_r = True
            if cost_r < best_cost:
                best_cost = cost_r
                best_p = p_r.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
