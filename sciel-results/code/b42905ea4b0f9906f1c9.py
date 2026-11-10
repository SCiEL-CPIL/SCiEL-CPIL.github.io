# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def prime_number_theorem_partial_function():
    """
    Maximizes a = -sum f(k) log(k)/k subject to M(x) <= 1.0001 for all x in [1, 10K].
    
    Strategy: Greedy coordinate descent.
    - Start with f(k) = 0 for all k in [1, K].
    - For each k (prioritizing high log(k)/k per unit constraint cost),
      try setting f(k) to -10 (lower bound). If M(x) stays <= 1.0001 for all x, accept.
    - After greedy pass, do local refinement: try adjusting f(k) values to push closer to bounds.
    - The S-adjustment on f(1) doesn't affect M(x) since floor(x/1)=x and the constraint
      is on M(x) = sum f(k)*floor(x/k). Wait - f(1) DOES affect M(x) since floor(x/1)=x.
      So we need to account for f(1) in the M(x) computation.
    """
    start_time = time.time()
    deadline = start_time + 590  # 40s safety margin before 630s kill
    
    def elapsed():
        return time.time() - start_time
    
    # Best fallback: seed-like solution
    best_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    best_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_score = 0.0
    
    def compute_S_and_adjust(domain, values):
        """Compute S = sum f(k)/k and return adjusted f values with f(1) adjusted."""
        values_clipped = np.clip(values, -10, 10)
        k_float = domain.astype(np.float64)
        S = np.sum(values_clipped / k_float)
        f_adj = values_clipped.copy()
        idx_1 = np.where(domain == 1)[0]
        if len(idx_1) > 0:
            f_adj[idx_1[0]] = f_adj[idx_1[0]] - S
        else:
            # If 1 not in domain, the evaluator treats f(1)=0 then replaces with 0-S=-S
            # This means effectively there's an implicit f(1) = -S contributing to M(x)
            # M(x) gets an extra term: (-S) * floor(x/1) = -S * x
            # We need to account for this. For safety, always include 1 in domain.
            pass
        return f_adj, S
    
    def check_M(domain, f_adj, K_max):
        """
        Check M(x) = sum_k f_adj(k) * floor(x/k) for x in [1, 10*K_max].
        Returns (max_M, feasible).
        """
        max_x = 10 * K_max
        if max_x <= 0:
            return 0.0, True
        
        # Efficient computation: for each k, floor(x/k) is piecewise constant
        # M(x) = sum_k f_adj[k] * floor(x/k)
        # We can compute this incrementally.
        
        # For memory efficiency with large K, use the fact that floor(x/k) changes
        # at multiples of k. We can use a difference array approach.
        
        # Alternative: direct computation but vectorized per k
        # For K up to ~5000, max_x = 50000, this is manageable.
        # For larger K, we need smarter approaches.
        
        if max_x <= 200000:
            # Direct vectorized approach
            x_vals = np.arange(1, max_x + 1, dtype=np.float64)
            M_sum = np.zeros(max_x, dtype=np.float64)
            
            for i in range(len(domain)):
                k = domain[i]
                val = f_adj[i]
                if abs(val) > 1e-12:
                    M_sum += val * np.floor(x_vals / k)
            
            max_M = np.max(M_sum)
            return max_M, (max_M <= 1.0001)
        else:
            # For very large K, use a smarter approach
            # floor(x/k) = n for x in [n*k, (n+1)*k - 1]
            # We can use a range-add approach with a difference array
            # But this is complex. For now, limit K to avoid this path.
            return float('inf'), False
    
    def compute_score(domain, values):
        """Compute the actual score as the evaluator would."""
        if len(domain) == 0:
            return 0.0, False
        
        f_adj, S = compute_S_and_adjust(domain, values)
        K_max = int(domain.max())
        
        max_M, feasible = check_M(domain, f_adj, K_max)
        if not feasible:
            return 0.0, False
        
        k_float = domain.astype(np.float64)
        a = -np.sum(f_adj * np.log(k_float) / k_float)
        return a, True
    
    # Strategy: Try various K values with greedy approach
    # The key insight: f(k) = -10 gives contribution 10*log(k)/k to objective
    # but costs constraint: for each x, it adds -10*floor(x/k) to M(x)
    # Since we want M(x) <= 1.0001, negative f(k) help (they decrease M(x))
    # Wait: M(x) = sum f(k)*floor(x/k). If f(k) is negative, M(x) decreases.
    # The constraint is M(x) <= 1.0001. So negative f(k) are good for feasibility.
    # But the objective is a = -sum f(k)*log(k)/k. If f(k) is negative, -f(k)*log(k)/k is positive.
    # So we want f(k) as negative as possible (-10) to maximize a.
    
    # But wait: the S-adjustment. f(1) gets adjusted by -S where S = sum f(k)/k.
    # If all f(k) for k>=2 are -10, then S = sum_{k>=2} (-10)/k = -10*H_{K-1} (roughly)
    # Then f(1) = 0 - S = 10*H_{K-1} which is large positive.
    # f(1) contributes f(1)*floor(x/1) = f(1)*x to M(x), which is very large positive!
    # This will violate M(x) <= 1.0001.
    
    # So the S-adjustment is critical. We need S to be close to 0 or negative
    # so that f(1) = -S is small or negative.
    
    # Actually, let me re-read: "f(1) is replaced by f(1) - S with S = sum_k f(k)/k"
    # So if we set f(1) = 0 initially, after adjustment f(1) = -S.
    # M(x) includes f(1)*x = -S*x.
    # For M(x) <= 1.0001, we need -S*x + sum_{k>=2} f(k)*floor(x/k) <= 1.0001.
    
    # If S is positive (sum f(k)/k > 0), then -S is negative, helping M(x) stay small.
    # If S is negative, -S is positive, hurting M(x).
    
    # To maximize a = -sum f(k)*log(k)/k:
    # - We want f(k) negative for k >= 2 (contributes positively to a)
    # - But this makes S negative (since f(k)/k < 0), making f(1) = -S positive
    # - f(1) positive hurts M(x)
    
    # Optimal tradeoff: choose f(k) for k>=2 to balance objective vs. M(x) constraint.
    
    # Greedy approach:
    # 1. Start with f(k) = 0 for all k.
    # 2. For k from K down to 2, try setting f(k) = -10. Check if M(x) stays feasible.
    # 3. After greedy pass, try to increase some f(k) back toward 0 if needed.
    
    # But the S-adjustment makes this tricky. Let's think differently.
    
    # Let g(k) = f(k) for k >= 2, and f(1) = 0 initially.
    # After S-adjustment: f_adj(1) = -S where S = sum_{k>=2} g(k)/k
    # f_adj(k) = g(k) for k >= 2
    # M(x) = -S*x + sum_{k>=2} g(k)*floor(x/k)
    #      = sum_{k>=2} g(k) * (floor(x/k) - x/k) + sum_{k>=2} g(k)*x/k - S*x
    #      = sum_{k>=2} g(k) * (floor(x/k) - x/k) + S*x - S*x
    #      = sum_{k>=2} g(k) * (floor(x/k) - x/k)
    
    # Wait, that's elegant! Let me verify:
    # M(x) = f_adj(1)*floor(x/1) + sum_{k>=2} f_adj(k)*floor(x/k)
    #      = (-S)*x + sum_{k>=2} g(k)*floor(x/k)
    # S = sum_{k>=2} g(k)/k
    # So M(x) = -x*sum_{k>=2} g(k)/k + sum_{k>=2} g(k)*floor(x/k)
    #         = sum_{k>=2} g(k) * (floor(x/k) - x/k)
    #         = sum_{k>=2} g(k) * (-(x/k - floor(x/k)))
    #         = -sum_{k>=2} g(k) * {x/k}
    
    # where {x/k} is the fractional part!
    
    # So M(x) = -sum_{k>=2} g(k) * {x/k}
    
    # The constraint M(x) <= 1.0001 becomes:
    # -sum_{k>=2} g(k) * {x/k} <= 1.0001
    # sum_{k>=2} g(k) * {x/k} >= -1.0001
    
    # Since {x/k} >= 0, if all g(k) >= 0, then sum >= 0 >= -1.0001, always feasible!
    # But we want g(k) negative to maximize a = -sum g(k)*log(k)/k.
    
    # If g(k) < 0, then sum_{k>=2} g(k)*{x/k} < 0, and we need it >= -1.0001.
    
    # Objective: a = -sum_{k>=2} g(k)*log(k)/k (since log(1)=0, f(1) doesn't contribute)
    # We want to maximize a, so minimize sum g(k)*log(k)/k, i.e., make g(k) as negative as possible.
    
    # Constraint: for all x in [1, 10K], sum_{k>=2} g(k)*{x/k} >= -1.0001
    # Equivalently: max_x sum_{k>=2} (-g(k))*{x/k} <= 1.0001
    # Let h(k) = -g(k) >= 0. Then:
    # max_x sum_{k>=2} h(k)*{x/k} <= 1.0001
    # Objective: a = sum_{k>=2} h(k)*log(k)/k
    # Bounds: h(k) in [0, 10]
    
    # This is the same LP as before but with the key insight that M(x) simplifies!
    
    # Now for greedy: we want to set h(k) as large as possible (up to 10)
    # while keeping max_x sum h(k)*{x/k} <= 1.0001.
    
    # The constraint cost of h(k) is max_x {x/k} weighted by h(k).
    # {x/k} ranges from 0 to almost 1.
    
    # Greedy: process k in order of highest log(k)/k per unit max fractional part.
    # Actually, just process k in descending order of log(k)/k.
    
    # For each k, the "cost" is h(k) * max_x {x/k}.
    # max_x {x/k} for x in [1, 10K] is close to 1 (when x = mk - 1 for some m).
    
    # So roughly, each unit of h(k) costs ~1 in the constraint.
    # Total constraint budget: 1.0001.
    # So we can have sum h(k) * (something close to 1) <= 1.0001.
    
    # This means we can only have a small total sum of h(k)!
    # If each h(k) costs ~1, we can only have sum h(k) ~ 1.
    
    # But h(k) can be fractional. So we set h(k) for a few k's.
    
    # Best strategy: set h(k) = 10 for the k with highest log(k)/k, but limited by constraint.
    # Or spread h(k) across many k's.
    
    # Let's do greedy: for k from 2 to K, try to increase h(k) by small increments
    # while maintaining feasibility.
    
    # Actually, for efficiency, let's try a simpler approach:
    # Set h(k) = c for a set of k's, where c is chosen to satisfy the constraint.
    
    # Or: set h(k) = 10 for k in a small set, and 0 otherwise.
    
    # Let me try: for each candidate K, do a greedy fill.
    
    K_candidates = [100, 500, 1000, 2000, 5000, 10000]
    
    for K in K_candidates:
        if elapsed() > deadline - 30:
            break
        
        # Greedy coordinate descent
        h = np.zeros(K - 1, dtype=np.float64)  # h[k-2] corresponds to k
        # Domain is [1, 2, ..., K], h corresponds to k=2..K
        
        # Try setting h(k) = 10 for each k, in order of decreasing log(k)/k
        k_order = np.arange(2, K + 1)
        # Sort by log(k)/k descending
        scores = np.array([math.log(k) / k for k in k_order])
        sort_idx = np.argsort(-scores)
        k_order_sorted = k_order[sort_idx]
        
        # For each k, try to set h(k) as high as possible
        # The constraint is: for all x in [1, 10K], sum_j h(j)*{x/j} <= 1.0001
        
        # To check feasibility efficiently, we can compute the max of the sum
        # But this is expensive for large K.
        
        # Alternative: use a probabilistic or approximate check.
        # Or: only check a subset of x values.
        
        # For now, let's try a simple approach:
        # Set h(k) = 10 for the first few k's, check feasibility, stop when infeasible.
        
        # Actually, let's do incremental:
        # Start with h = 0. For each k in sorted order, try h(k) = 10.
        # If infeasible, try h(k) = 5, 2.5, 1, 0.5, 0.25, 0.1, 0.
        # Accept the largest feasible value.
        
        max_x = 10 * K
        # Precompute fractional parts for efficiency?
        # {x/k} for x in [1, max_x], k in [2, K]
        # This is a max_x x (K-1) matrix, which can be large.
        
        # For K=10000, max_x=100000, matrix is 100000 x 9999 ~ 10^9 elements. Too large.
        
        # Need a smarter approach.
        
        # Insight: {x/k} = x/k - floor(x/k). The sum sum_k h(k)*{x/k} 
        # can be computed as sum_k h(k)*x/k - sum_k h(k)*floor(x/k).
        # But this doesn't help much.
        
        # Alternative: for each x, the constraint is linear in h.
        # We can use a lazy approach: only check a few x values.
        
        # Key x values: x = 1, 2, ..., and x = mk - 1 for various m, k.
        # The worst case for {x/k} is when x/k is just below an integer.
        
        # Let's try: check x = 1, 2, ..., min(100, max_x) and x = k-1 for k in [2, K].
        
        # Actually, for a given h, the max of sum h(k)*{x/k} over x is hard to compute.
        # Let's just check all x for small K, and sample for large K.
        
        def is_feasible(h, K, sample=False):
            """Check if sum_k h(k)*{x/k} <= 1.0001 for all x in [1, 10K]."""
            max_x = 10 * K
            if max_x <= 50000:
                # Check all x
                x_vals = np.arange(1, max_x + 1, dtype=np.float64)
                # Compute sum_k h(k)*{x/k} for all x
                # h is indexed by k-2 for k in [2, K]
                total = np.zeros(max_x, dtype=np.float64)
                for j in range(len(h)):
                    if h[j] != 0:
                        k = j + 2
                        frac = (x_vals / k) - np.floor(x_vals / k)
                        total += h[j] * frac
                return np.max(total) <= 1.0001
            else:
                # Sample x values
                if sample:
                    x_samples = np.random.choice(max_x, size=min(10000, max_x), replace=False) + 1
                else:
                    x_samples = np.arange(1, min(1000, max_x) + 1)
                total = 0.0
                for j in range(len(h)):
                    if h[j] != 0:
                        k = j + 2
                        frac = (x_samples / k) - np.floor(x_samples / k)
                        total += h[j] * frac
                return np.max(total) <= 1.0001
        
        # Greedy fill
        for idx in sort_idx:
            if elapsed() > deadline - 20:
                break
            k = idx + 2
            # Try decreasing values of h(k)
            for trial_h in [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01]:
                h[idx] = trial_h
                if is_feasible(h, K):
                    break
                h[idx] = 0.0
        
        # Construct domain and values
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = np.zeros(K, dtype=np.float64)
        values[0] = 0.0  # f(1) = 0
        values[1:] = -h  # f(k) = -h(k)
        
        score, feasible = compute_score(domain, values)
        if feasible and score > best_score:
            best_score = score
            best_domain = domain.copy()
            best_values = values.copy()
    
    # Also try the seed solution
    score_seed, feas_seed = compute_score(best_domain, best_values) if best_score > 0 else (0.0, False)
    
    # Final: return best found
    return best_domain, best_values

# EVOLVE-BLOCK-END
