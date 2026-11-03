import numpy as np
import time

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    def compute_cost(p):
        # p is a permutation of 0..n-1
        # Cost = sum a[i,j] * b[p[i], p[j]]
        # Vectorized: (a * b[np.ix_(p, p)]).sum()
        return np.sum(a * b[np.ix_(p, p)])

    def local_search(p, time_limit):
        p = p.copy()
        current_cost = compute_cost(p)
        
        # Precompute row sums of a? No, we need specific differences.
        # We will iterate through all pairs.
        # To avoid O(n^3) total time per pass (n^2 pairs * O(n) cost), 
        # we can just do it. n=100, n^2=10000, n=100. 10000*100 = 1e6 ops per pass.
        # Python loops might be slow. 1e6 ops * ~10 restarts * ~10 passes = 1e8 ops.
        # Might be tight. Let's optimize the inner loop.
        
        improved = True
        while improved:
            improved = False
            start_time = time.time()
            for i in range(n):
                for j in range(i + 1, n):
                    if time.time() - start_time > 5: # Check time periodically
                        return p, current_cost
                    
                    u = p[i]
                    v = p[j]
                    
                    # Delta calculation
                    # delta = (a[i,i] + a[j,j]) * (b[v,v] - b[u,u])
                    #       + sum_{k != i,j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
                    
                    # We can compute the sum part using numpy
                    # diff_a = a[i] - a[j]
                    # diff_b = b[v, :] - b[u, :]
                    # sum_part = np.sum(diff_a * diff_b)
                    # We need to exclude k=i and k=j.
                    # term_i = diff_a[i] * diff_b[u]  (since p[i]=u)
                    # term_j = diff_a[j] * diff_b[v]  (since p[j]=v)
                    # Note: diff_b[u] is b[v,u] - b[u,u]. diff_b[v] is b[v,v] - b[u,v].
                    
                    diff_a = a[i] - a[j]
                    diff_b = b[v, :] - b[u, :]
                    
                    # The sum over ALL k is np.dot(diff_a, diff_b)
                    # We subtract the terms for k=i and k=j
                    # For k=i: a[i,i]-a[j,i] * (b[v,p[i]] - b[u,p[i]]) = (a[i,i]-a[j,i]) * (b[v,u] - b[u,u])
                    # For k=j: a[i,j]-a[j,j] * (b[v,p[j]] - b[u,p[j]]) = (a[i,j]-a[j,j]) * (b[v,v] - b[u,v])
                    
                    total_dot = np.dot(diff_a, diff_b)
                    term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                    term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                    
                    sum_part = total_dot - term_i - term_j
                    
                    delta = (a[i, i] + a[j, j]) * (b[v, v] - b[u, u]) + sum_part
                    
                    if delta < -1e-9: # Improvement
                        p[i], p[j] = v, u
                        current_cost += delta
                        improved = True
                        # Optional: restart scan or continue. Standard 2-opt continues.
                        # For speed, we might want to break and restart if we want to be aggressive,
                        # but continuing is fine.
        
        return p, current_cost

    best_p = None
    best_cost = float('inf')
    
    start_time = time.time()
    time_limit = 100.0 # Safety margin for 120s limit
    
    # Initial heuristic: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p0 = np.empty(n, dtype=np.int64)
    p0[heavy] = central
    
    # Run local search on the initial heuristic
    p_sol, cost_sol = local_search(p0, time_limit)
    if cost_sol < best_cost:
        best_cost = cost_sol
        best_p = p_sol.copy()
        
    elapsed = time.time() - start_time
    
    # If we have time, do random restarts
    rng = np.random.default_rng(42)
    while elapsed < time_limit:
        # Random initialization
        p_rand = rng.permutation(n)
        
        # Maybe perturb the best solution found so far?
        # Let's stick to random for diversity.
        
        p_sol, cost_sol = local_search(p_rand, time_limit)
        if cost_sol < best_cost:
            best_cost = cost_sol
            best_p = p_sol.copy()
        
        elapsed = time.time() - start_time
        
        # If we found a very good solution, we might want to stop earlier?
        # Or just let it run out of time.
        
    return best_p.tolist()
