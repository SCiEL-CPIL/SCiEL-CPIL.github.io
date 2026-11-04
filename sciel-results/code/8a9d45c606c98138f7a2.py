"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    """
    Solve QAP instance: assign facilities to locations to minimize sum a[i,j]*b[p[i],p[j]].
    Uses multi-start local search with 2-opt swaps.
    """
    n = len(a)
    start_time = time.time()
    deadline = 145.0  # safety margin of 5s

    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def local_search(p, max_iters=5000):
        """Improve permutation using 2-opt swaps."""
        p = np.array(p, dtype=np.int64, copy=True)
        current_cost = compute_cost(p)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try random swaps
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                i, j = np.random.choice(n, 2, replace=False)
                
                # Compute new cost after swap
                p[i], p[j] = p[j], p[i]
                new_cost = compute_cost(p)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    break
                else:
                    p[i], p[j] = p[j], p[i]
            
            if not improved:
                break
        
        return p, current_cost

    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    best_p = p.copy()
    best_cost = compute_cost(p)
    
    # Local search from initial
    p_ls, cost_ls = local_search(p)
    if cost_ls < best_cost:
        best_p = p_ls.copy()
        best_cost = cost_ls
    
    # Multi-start
    np.random.seed(42)
    while time.time() < deadline:
        p_rand = np.random.permutation(n)
        p_ls, cost_ls = local_search(p_rand)
        if cost_ls < best_cost:
            best_p = p_ls.copy()
            best_cost = cost_ls
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
