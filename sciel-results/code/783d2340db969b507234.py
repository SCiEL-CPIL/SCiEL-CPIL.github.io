"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    def total_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    t0 = time.time()
    time_limit = 110.0
    rng = np.random.default_rng(42)
    
    best_cost = np.iinfo(np.int64).max
    best_p = None

    while time.time() - t0 < time_limit:
        # 1. Constructive Phase: Greedy with Randomization
        # Calculate flow sums
        flow_sums = a.sum(axis=1) + a.sum(axis=0)
        # Get top K facilities to consider in RCL
        rcl_size = max(10, int(n * 0.1))
        
        p = np.full(n, -1, dtype=np.int64)
        used = np.zeros(n, dtype=bool)
        
        # We will assign facilities one by one
        # To diversify, we pick the next facility from the top flow_sums 
        # but with some randomness
        
        # Simple Greedy Construction with Random Location Selection
        # Sort facilities by flow sum descending
        order = np.argsort(-flow_sums, kind="stable")
        
        for idx in range(n):
            f = order[idx]
            # Calculate incremental cost for each unused location
            # inc = sum_{g assigned} (a[f, g] * b[loc, p[g]] + a[g, f] * b[p[g], loc])
            inc_costs = np.zeros(n, dtype=np.int64)
            for k in range(idx):
                g = order[k]
                loc_g = p[g]
                # Vectorized update for all locations
                # b[:, loc_g] is the column for loc_g
                # b[loc_g, :] is the row for loc_g
                inc_costs += a[f, g] * b[:, loc_g] + a[g, f] * b[loc_g, :]
            
            # Mask used locations
            inc_costs[used] = np.iinfo(np.int64).max
            
            # Add some noise to diversify
            noise = rng.integers(0, 1000, size=n)
            # Scale noise so it doesn't dominate, but allows different choices
            # If cost is large, noise should be small relative to cost differences
            # Simple approach: just add small noise
            inc_costs += noise
            
            # Pick best location
            best_loc = np.argmin(inc_costs)
            p[f] = best_loc
            used[best_loc] = True

        current_cost = total_cost(p)
        
        # 2. Local Search Phase: 2-opt First Improvement
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if improved:
                    break
                for j in range(i + 1, n):
                    u = p[i]
                    v = p[j]
                    
                    # Compute delta
                    # delta = sum_{k != i, j} [(a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]]) + (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])]
                    #       + (a[i,j]-a[j,i])*(b[v,u]-b[u,v])
                    
                    # Vectorized calculation
                    # diff_row = a[i, :] - a[j, :]
                    # diff_col = a[:, i] - a[:, j]
                    # bv_p = b[v, p]
                    # bu_p = b[u, p]
                    # bp_v = b[p, v]
                    # bp_u = b[p, u]
                    
                    # This is O(n) per pair.
                    # To speed up, we can skip if a[i,j] and a[j,i] are 0 and a[i,:] == a[j,:]?
                    # No, just compute.
                    
                    diff_row = a[i, :] - a[j, :]
                    diff_col = a[:, i] - a[:, j]
                    
                    bv_p = b[v, p]
                    bu_p = b[u, p]
                    bp_v = b[p, v]
                    bp_u = b[p, u]
                    
                    s1 = np.dot(diff_row, bv_p - bu_p)
                    s2 = np.dot(diff_col, bp_v - bp_u)
                    
                    # Subtract k=i and k=j terms
                    s1 -= diff_row[i] * (b[v, u] - b[u, u])
                    s2 -= diff_col[i] * (b[u, v] - b[u, u])
                    
                    s1 -= diff_row[j] * (b[v, v] - b[u, v])
                    s2 -= diff_col[j] * (b[v, v] - b[v, u])
                    
                    direct = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
                    
                    delta = s1 + s2 + direct
                    
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
                        break
        
        # 3. Update Best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
