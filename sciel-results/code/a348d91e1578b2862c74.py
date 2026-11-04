import numpy as np
import time

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Rank Matching
    # Heavy flows (high row sum in A) to short distances (low row sum in B)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Calculate initial cost
    # Cost = sum a[i,j] * b[p[i], p[j]]
    # This is effectively trace(A * B_p) where B_p is permuted B
    # But calculating full cost is O(n^2).
    # We only need it for tracking best.
    def calc_cost(p):
        # Vectorized cost calculation
        # B_perm[i, j] = b[p[i], p[j]]
        # Cost = sum(a * B_perm)
        # We can do this as:
        # For each i, sum_j a[i,j] * b[p[i], p[j]]
        # This is row-wise dot product of a[i] with b[p[i], :]
        # Using np.einsum or broadcasting
        # B_perm = b[np.ix_(p, p)]
        # return np.sum(a * B_perm)
        # This creates a large intermediate array (n x n). n=256, 65k elements. Fine.
        B_perm = b[np.ix_(p, p)]
        return np.sum(a * B_perm)

    current_cost = calc_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Simulated Annealing
    # Estimate iterations
    # Let's aim for ~5 million iterations
    # T_start: estimate typical delta. 
    # Random swap delta magnitude?
    # Let's just pick a heuristic T_start.
    # Or compute max possible delta?
    # Simple heuristic: T_start = 10 * max(a) * max(b) ? No, that's too big.
    # Let's try T_start = 1000.0 and see.
    # Better: Compute average absolute delta of random swaps?
    # Too slow to do many random swaps just to estimate.
    # Let's use a fixed schedule.
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s for final polish and overhead
    
    # Parameters
    # We don't know exact iteration speed, so we adapt T?
    # Or just run until time is up.
    # If we use geometric cooling, we need to know N_iters.
    # Instead, let's use a "time-based" temperature decay or just a very slow decay.
    # T = T_init * (decay_factor)^(iter)
    # If we run ~5M iters, and want T to drop by factor 1000:
    # 1000 = decay^5M -> decay = 1000^(-1/5M) ~ 0.9999986
    # This is hard to tune blindly.
    
    # Alternative: Run SA for fixed number of iterations, then check time.
    # Or, use a simple rule:
    # T starts at 1.0? No, costs are large (44M).
    # Deltas can be large.
    # Let's estimate delta magnitude.
    # Max element in A ~ ? Max element in B ~ ?
    # For tai256c, values are integers.
    # Let's check max values to set T.
    max_a = np.max(a)
    max_b = np.max(b)
    # Typical delta is roughly n * max_a * max_b? No, it's a sum of n terms.
    # Term is a[i,k]*(b[v,pk]-b[u,pk]).
    # If a ~ 100, b ~ 1000, diff ~ 1000. Sum of 256 terms ~ 25M.
    # So T should start around 10M - 50M to accept random moves.
    
    T = 5000000.0
    T_min = 1.0
    # We will decrease T linearly or geometrically.
    # Let's use geometric.
    # We don't know total iterations.
    # Let's just run while time < limit.
    # To ensure convergence, we can decrease T every K iterations.
    # Or, simpler:
    # T = T_init * exp(-lambda * t_elapsed)
    # Let's try this.
    
    # Precompute for speed
    # We need a[i] and a[j] quickly.
    # We need b[v, p] and b[u, p] quickly.
    
    rng = np.random.default_rng(42)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Select random i, j
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
            
        u = p[i]
        v = p[j]
        
        # Calculate Delta
        # diff_a = a[i] - a[j]
        # diff_b = b[v, p] - b[u, p]
        # S = dot(diff_a, diff_b)
        # term_i = diff_a[i] * diff_b[i]
        # term_j = diff_a[j] * diff_b[j]
        # delta = S - term_i - term_j
        
        # Optimized:
        diff_a = a[i] - a[j]
        # b[v, p] is a row of b indexed by p
        # We can precompute B_rows? No, p changes.
        # b[v, p] creates a copy.
        # b[u, p] creates a copy.
        # This is O(n) copy + O(n) sub + O(n) dot.
        
        # To reduce copies:
        # We can use b[v, :] and index? No, p is the permutation.
        # b[v, p] is necessary.
        
        diff_b = b[v, p] - b[u, p]
        
        S = np.dot(diff_a, diff_b)
        
        # Subtract terms for k=i and k=j
        # Note: p[i] is u, p[j] is v.
        # diff_b[i] = b[v, p[i]] - b[u, p[i]] = b[v, u] - b[u, u]
        # diff_b[j] = b[v, p[j]] - b[u, p[j]] = b[v, v] - b[u, v]
        
        term_i = diff_a[i] * diff_b[i]
        term_j = diff_a[j] * diff_b[j]
        
        delta = S - term_i - term_j
        
        # Accept?
        if delta < 0:
            # Improve
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # Worse, accept with probability exp(-delta/T)
            # T decay
            # Simple time-based decay:
            # T = T_init * (T_end/T_init)^(elapsed/total_time)
            # Let's use a fixed decay rate per second?
            # T = T_init * exp(-cooling_rate * elapsed)
            # If T_init = 5e6, and we want T ~ 1 at 115s:
            # 1 = 5e6 * exp(-rate * 115)
            # exp(rate*115) = 5e6
            # rate * 115 = ln(5e6) ~ 15.4
            # rate ~ 0.134
            T = 5000000.0 * np.exp(-0.134 * elapsed)
            
            if T > T_min:
                if rng.random() < np.exp(-delta / T):
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
            else:
                # T is effectively 0, stop accepting worse moves
                # We could break here or just stop accepting
                pass

    # Final Polish: Best Improvement Local Search
    # Try all pairs (i, j) and swap if it improves.
    # Repeat until no improvement? Or just one pass?
    # One pass is O(n^2) swaps.
    # Given time constraints, one pass is good.
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                u = p[i]
                v = p[j]
                diff_a = a[i] - a[j]
                diff_b = b[v, p] - b[u, p]
                S = np.dot(diff_a, diff_b)
                term_i = diff_a[i] * diff_b[i]
                term_j = diff_a[j] * diff_b[j]
                delta = S - term_i - term_j
                
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
        
        # Check time in polish loop?
        if time.time() - start_time > time_limit:
            break

    return best_p.tolist()
