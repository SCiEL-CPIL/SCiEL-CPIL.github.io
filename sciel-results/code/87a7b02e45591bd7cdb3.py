import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Heuristic Rank Matching
    # Assign facilities with high flow to locations with low distance sum (central)
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    fac_order = np.argsort(-flow_sums, kind="stable")
    loc_order = np.argsort(dist_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order
    
    # Precompute B_perm: B_perm[x, j] = b[x, p[j]]
    B_perm = b[:, p]
    
    # Calculate initial cost
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # b[p[i], p[j]] = B_perm[p[i], j]
    # We can compute this as np.sum(a * B_perm.T) ?
    # a[i,j] * B_perm[p[i], j].
    # Let's just use the safe method for init.
    cost = np.sum(a * b[np.ix_(p, p)])
    
    best_p = p.copy()
    best_cost = cost
    
    start_time = time.time()
    time_limit = 115  # Leave buffer for overhead
    
    # Iterated Local Search
    while time.time() - start_time < time_limit:
        improved = True
        # Local Search Phase: 2-opt
        while improved:
            improved = False
            # Iterate through all pairs
            for i in range(n):
                for k in range(i + 1, n):
                    u = p[i]
                    v = p[k]
                    
                    # Delta calculation using B_perm
                    diff_a = a[i] - a[k]
                    diff_b = B_perm[v] - B_perm[u]
                    S = np.dot(diff_a, diff_b)
                    
                    # Subtract self and cross terms included in S
                    S -= (a[i, i] - a[k, i]) * (b[v, u] - b[u, u])
                    S -= (a[i, k] - a[k, k]) * (b[v, v] - b[u, v])
                    
                    # Add correct new terms for i, k interaction and self loops
                    delta = S + (a[i, k] * (b[v, u] - b[u, v]) + 
                                 a[k, i] * (b[u, v] - b[v, u]) + 
                                 a[i, i] * (b[v, v] - b[u, u]) + 
                                 a[k, k] * (b[u, u] - b[v, v]))
                    
                    if delta < 0:
                        p[i], p[k] = p[k], p[i]
                        # Swap columns i and k in B_perm
                        B_perm[:, [i, k]] = B_perm[:, [k, i]]
                        cost += delta
                        improved = True
                        if cost < best_cost:
                            best_cost = cost
                            best_p = p.copy()
        
        # Perturbation Phase
        # Perform a few random swaps to escape local optimum
        for _ in range(5):
            i = random.randint(0, n - 1)
            k = random.randint(0, n - 1)
            if i != k:
                p[i], p[k] = p[k], p[i]
                B_perm[:, [i, k]] = B_perm[:, [k, i]]
        
        # Recalculate cost from scratch to avoid accumulation errors
        cost = np.sum(a * b[np.ix_(p, p)])
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()

    return best_p.tolist()
