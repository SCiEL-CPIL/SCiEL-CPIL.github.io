"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s kill

    # 1. Initialization: Rank-matching (Heavy flows to Central locations)
    # This is a fast, reasonable start.
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    best_cost = compute_cost(p)
    best_p = p.copy()

    # Precompute transpose of a for column access if needed, 
    # though a[:, i] is a view and fast.
    
    def swap_delta(i, j, p, a, b):
        u = p[i]
        v = p[j]
        
        # Vectorized delta calculation
        # delta = sum_k (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]]) + sum_k (a[k,i]-a[k,j])*(b[p[k],v]-b[p,k,u])
        
        # Row differences
        diff_row = b[v, p] - b[u, p]
        term1 = np.dot(a[i, :] - a[j, :], diff_row)
        
        # Col differences
        diff_col = b[p, v] - b[p, u]
        term2 = np.dot(a[:, i] - a[:, j], diff_col)
        
        return term1 + term2

    # 2. Local Search: First-Improvement 2-opt
    # This is much faster per pass, allowing many more passes within the time budget.
    
    def local_search(p, current_cost, deadline):
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(n - 1):
                if time.time() >= deadline - 1.0:
                    return p, current_cost
                for j in range(i + 1, n):
                    if time.time() >= deadline - 1.0:
                        return p, current_cost
                    d = swap_delta(i, j, p, a, b)
                    if d < 0:
                        # Apply the move immediately
                        p[i], p[j] = p[j], p[i]
                        current_cost += d
                        improved = True
                        break
                if improved:
                    break
                
        return p, current_cost

    # Run local search
    p, current_cost = local_search(p, best_cost, deadline)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()

    # 3. Restart with Perturbation if time remains
    # This helps escape local minima.
    while time.time() < deadline - 2.0: # Keep 2s safety margin for final checks
        # Perturb the current best solution
        # Randomly swap a few pairs
        p = best_p.copy()
        current_cost = best_cost
        
        # Number of swaps to perturb: maybe 5-10
        num_swaps = np.random.randint(5, 11)
        for _ in range(num_swaps):
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        # Run local search from perturbed state
        p, current_cost = local_search(p, current_cost, deadline)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    # Final verification
    final_cost = compute_cost(best_p)
    if final_cost < best_cost:
        best_cost = final_cost

    return best_p.tolist()
# EVOLVE-BLOCK-END
