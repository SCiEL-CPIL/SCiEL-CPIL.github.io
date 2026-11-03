"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Rank-matching start: rows of the first matrix with the largest totals go to the
    # rows of the second matrix with the smallest totals (heavy flows on short distances).
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Compute initial cost
    current_cost = int(np.sum(a * b[np.ix_(p, p)]))
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute row sums of a for quick delta calculations
    a_row_sums = a.sum(axis=1)
    
    # Local search: 2-opt style swaps
    while time.time() < deadline:
        improved = False
        # Try all pairs of swaps, but limit to a subset if time is running out
        # For n=100, there are 4950 pairs, which is manageable
        for i in range(n):
            if time.time() > deadline:
                break
            pi = p[i]
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                pj = p[j]
                
                # Calculate delta for swapping facilities at locations pi and pj
                # Cost change = sum over all k of:
                #   a[i,k] * (b[pj,p[k]] - b[pi,p[k]]) + 
                #   a[k,i] * (b[p[k],pj] - b[p[k],pi]) +
                #   a[j,k] * (b[pi,p[k]] - b[pj,p[k]]) +
                #   a[k,j] * (b[p[k],pi] - b[p[k],pj]) +
                #   (a[i,j] + a[j,i]) * (b[pi,pj] - b[pj,pi])
                # But simpler: just compute the difference directly
                
                # Swap pi and pj in p temporarily
                p[i], p[j] = p[j], p[i]
                new_cost = int(np.sum(a * b[np.ix_(p, p)]))
                delta = new_cost - current_cost
                
                if delta < 0:
                    current_cost = new_cost
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                else:
                    # Swap back
                    p[i], p[j] = p[j], p[i]
        
        if not improved:
            break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
