"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    # Safety margin: 120s limit, aim to finish by 115s
    deadline = time.time() + 115.0

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def swap_delta(i, j, p):
        u = p[i]
        v = p[j]
        
        # Vectorized delta calculation
        # S1 = sum_k (a[i, k] - a[j, k]) * (b[v, p[k]] - b[u, p[k]])
        # S2 = sum_k (a[k, i] - a[k, j]) * (b[p[k], v] - b[p[k], u])
        
        diff_a_rows = a[i, :] - a[j, :]
        diff_b_rows = b[v, p] - b[u, p]
        s1 = np.dot(diff_a_rows, diff_b_rows)
        
        diff_a_cols = a[:, i] - a[:, j]
        diff_b_cols = b[p, v] - b[p, u]
        s2 = np.dot(diff_a_cols, diff_b_cols)
        
        # Correction for k=i and k=j
        # Terms in s1 for k=i: (a[i,i] - a[j,i]) * (b[v, u] - b[u, u])
        # Terms in s2 for k=i: (a[i,i] - a[j,i]) * (b[u, v] - b[u, u])
        # Terms in s1 for k=j: (a[i,j] - a[j,j]) * (b[v, v] - b[u, v])
        # Terms in s2 for k=j: (a[i,j] - a[j,j]) * (b[u, v] - b[v, v])
        
        ai_i = a[i, i]
        aj_i = a[j, i]
        ai_j = a[i, j]
        aj_j = a[j, j]
        
        bu_u = b[u, u]
        bu_v = b[u, v]
        bv_u = b[v, u]
        bv_v = b[v, v]
        
        corr_1 = (ai_i - aj_i) * (bv_u - bu_u)
        corr_2 = (ai_i - aj_i) * (bu_v - bu_u)
        corr_3 = (ai_j - aj_j) * (bv_v - bu_v)
        corr_4 = (ai_j - aj_j) * (bu_v - bv_v)
        
        # Correct interaction term for i, j
        # New interaction: a[i,j]b[v,u] + a[j,i]b[u,v]
        # Old interaction: a[i,j]b[u,v] + a[j,i]b[v,u]
        # Delta interaction: (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
        # Wait, a and b are not necessarily symmetric.
        # Old: a_ij b_uv + a_ji b_vu
        # New: a_ij b_vu + a_ji b_uv
        # Diff: a_ij(b_vu - b_uv) + a_ji(b_uv - b_vu) = (a_ij - a_ji)(b_vu - b_uv)
        corr_interaction = (ai_j - aj_i) * (bv_u - bu_v)
        
        return s1 + s2 - (corr_1 + corr_2 + corr_3 + corr_4) + corr_interaction

    def local_search(p, current_cost, deadline):
        improved = True
        # Limit iterations to prevent infinite loops, though steepest descent converges quickly
        max_iter = 500 
        iter_count = 0
        
        while improved and time.time() < deadline - 2.0 and iter_count < max_iter:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Scan all pairs to find the steepest descent move
            for i in range(n - 1):
                if time.time() >= deadline - 2.0:
                    return p, current_cost
                for j in range(i + 1, n):
                    if time.time() >= deadline - 2.0:
                        return p, current_cost
                    d = swap_delta(i, j, p)
                    if d < best_delta:
                        best_delta = d
                        best_i = i
                        best_j = j
            
            if best_delta < 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
                iter_count += 1
            else:
                break
                
        return p, current_cost

    def greedy_init():
        """Greedy construction: assign facilities in order of flow sum to best available location."""
        p = np.full(n, -1, dtype=np.int64)
        flow_sums = a.sum(axis=1)
        order = np.argsort(-flow_sums, kind='stable')
        
        # To speed up greedy, we can just use the rank matching as a base and improve
        # But let's try a proper greedy for one of the initializations
        assigned = np.zeros(n, dtype=bool)
        current_cost = 0
        
        for idx in order:
            best_loc = -1
            best_inc = np.inf
            for loc in range(n):
                if assigned[loc]:
                    continue
                # Incremental cost of placing idx at loc
                # Sum over already assigned facilities j
                # (a[idx, j] + a[j, idx]) * b[loc, p[j]]
                # Vectorized:
                # mask = assigned
                # inc = np.dot((a[idx, :] + a[:, idx]) * mask, b[loc, p[mask]])
                # This is O(N) per location, O(N^2) per facility, O(N^3) total.
                # N=150, N^3 = 3.3M. This is very fast.
                
                # Optimization: Precompute row sums?
                # Let's just do it directly.
                inc = 0
                # Using numpy for the dot product part
                # We need b[loc, p[assigned]]
                # And (a[idx, assigned] + a[assigned, idx])
                
                # Since assigned is a mask, we can't easily index p with it without creating arrays.
                # Let's stick to rank-matching for speed, it's usually good enough for initialization.
                pass
            
            # Fallback to rank matching for speed in initialization
            # The greedy is too slow in Python loops if not carefully vectorized.
            # Let's use rank matching for all initializations, but vary the permutation slightly.
            pass
            
        return None # Not used

    # 1. Initialization: Rank-matching
    # Assign facilities with highest flow sum to locations with lowest distance sum
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    
    best_cost = np.inf
    best_p = None
    
    # Try a few different initializations
    # 1. Standard rank matching
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central
    
    # 2. Reversed rank matching (sometimes helps)
    p_init2 = np.empty(n, dtype=np.int64)
    p_init2[heavy] = central[::-1]
    
    # 3. Random
    p_init3 = np.random.permutation(n)
    
    inits = [p_init, p_init2, p_init3]
    
    for p in inits:
        if time.time() > deadline - 5.0:
            break
        cost = compute_cost(p)
        p, cost = local_search(p, cost, deadline)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            
    # 3. Iterated Local Search (ILS)
    # Use random perturbations to escape local minima
    
    # Set a safety margin for the final return
    while time.time() < deadline - 10.0: 
        # Perturb the best solution
        p = best_p.copy()
        current_cost = best_cost
        
        # Perform random swaps
        # Number of swaps: typically between 5 and 15 for n=150
        # Adaptive perturbation size?
        num_swaps = np.random.randint(5, 16)
        for _ in range(num_swaps):
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        # Compute cost of perturbed solution
        current_cost = compute_cost(p)
        
        # Run local search from the perturbed state
        p, current_cost = local_search(p, current_cost, deadline)
        
        # Update best if improved
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # If time is running out, we might want to stop early to ensure clean exit
        if time.time() > deadline - 15.0:
            break

    # Final verification
    final_cost = compute_cost(best_p)
    if final_cost < best_cost:
        best_cost = final_cost

    return best_p.tolist()

# EVOLVE-BLOCK-END
