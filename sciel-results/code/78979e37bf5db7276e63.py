"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        return int(np.sum(a * b[np.ix_(p, p)]))

    def compute_swap_delta(i, j, p, a, b, n):
        """Compute cost delta when swapping locations of facilities i and j."""
        li = p[i]
        lj = p[j]
        
        if li == lj:
            return 0
        
        # Precompute differences for vectorized computation
        # delta = sum over all terms that change
        
        # Terms involving i or j:
        # For k not in {i,j}:
        #   a[i,k]*(b[lj,p[k]] - b[li,p[k]]) + a[k,i]*(b[p[k],lj] - b[p[k],li])
        #   + a[j,k]*(b[li,p[k]] - b[lj,p[k]]) + a[k,j]*(b[p[k],li] - b[p[k],lj])
        
        # Direct computation for n=150 is fast enough
        delta = 0
        
        # Diagonal terms
        delta += a[i,i] * (b[lj,lj] - b[li,li])
        delta += a[j,j] * (b[li,li] - b[lj,lj])
        
        # Cross terms between i and j
        delta += a[i,j] * (b[lj,li] - b[li,lj])
        delta += a[j,i] * (b[li,lj] - b[lj,li])
        
        # Terms with other facilities
        for k in range(n):
            if k == i or k == j:
                continue
            pk = p[k]
            delta += a[i,k] * (b[lj,pk] - b[li,pk])
            delta += a[k,i] * (b[pk,lj] - b[pk,li])
            delta += a[j,k] * (b[li,pk] - b[lj,pk])
            delta += a[k,j] * (b[pk,li] - b[pk,lj])
        
        return delta

    def local_search(p, a, b, n, deadline, time_budget=0.5):
        """Perform steepest-descent 2-swap local search."""
        cost = compute_cost(p)
        improved = True
        iterations = 0
        
        while improved and time.time() < deadline - time_budget:
            improved = False
            iterations += 1
            best_delta = 0
            best_i, best_j = -1, -1
            
            for i in range(n):
                for j in range(i+1, n):
                    delta = compute_swap_delta(i, j, p, a, b, n)
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            
            if best_i >= 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cost += best_delta
                improved = True
        
        return p, cost

    def perturb(p, a, b, n, rng, strength=10):
        """Perturb permutation by random swaps."""
        p_new = p.copy()
        for _ in range(strength):
            i, j = rng.choice(n, size=2, replace=False)
            p_new[i], p_new[j] = p_new[j], p_new[i]
        return p_new

    # Initial heuristic: rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    best_p = p.copy()
    best_cost = compute_cost(p)
    
    # Run ILS (Iterated Local Search)
    rng = np.random.RandomState(42)
    ilp = p.copy()
    ilp_cost = best_cost
    
    while time.time() < deadline - 2:
        # Local search from current position
        ilp, ilp_cost = local_search(ilp, a, b, n, deadline, time_budget=2)
        
        if ilp_cost < best_cost:
            best_p = ilp.copy()
            best_cost = ilp_cost
        
        # Perturb for next iteration
        if time.time() < deadline - 5:
            ilp = perturb(ilp, a, b, n, rng, strength=8)
    
    # Try a few different initializations if time permits
    if time.time() < deadline - 10:
        # Inverse rank matching
        heavy2 = np.argsort(a.sum(axis=1), kind="stable")
        central2 = np.argsort(-b.sum(axis=1), kind="stable")
        p2 = np.empty(n, dtype=np.int64)
        p2[heavy2] = central2
        
        p2, cost2 = local_search(p2, a, b, n, deadline, time_budget=5)
        if cost2 < best_cost:
            best_p = p2.copy()
            best_cost = cost2
    
    if time.time() < deadline - 10:
        # Random start
        p3 = rng.permutation(n).astype(np.int64)
        p3, cost3 = local_search(p3, a, b, n, deadline, time_budget=5)
        if cost3 < best_cost:
            best_p = p3.copy()
            best_cost = cost3
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
