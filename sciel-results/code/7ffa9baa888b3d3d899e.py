import numpy as np
import time
import sys

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 600.0
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Strategy: 
    # The objective is to maximize a = -sum_{k=2}^K g(k) log(k)/k.
    # Since log(k)/k > 0 for k >= 2, we want g(k) to be as negative as possible (-10) 
    # for small k (where log(k)/k is large) and potentially positive for large k 
    # if it helps satisfy the constraint M(x) <= 1.
    # 
    # The constraint is M(x) = sum_{k=1}^K f(k) floor(x/k) <= 1 for all x in [1, 10K].
    # With f(1) adjustment: M(x) = sum_{k=2}^K g(k) (floor(x/k) - x/k) <= 1.
    # Let A(x, k) = floor(x/k) - x/k. Note A(x, k) <= 0.
    # So the constraint is sum_{k=2}^K g(k) A(x, k) <= 1.
    # Since A(x, k) is negative, if we set g(k) = -10, the term is 10 * |A(x, k)|, which adds to the sum.
    # If we set g(k) = 10, the term is -10 * |A(x, k)|, which subtracts from the sum (helps feasibility).
    
    # We suspect the optimal solution involves setting g(k) = -10 for a small set of "critical" small k's,
    # and g(k) = 0 for most others, or g(k) = 10 for some large k's to balance.
    # However, setting g(k) = 10 for large k has a small negative contribution to the objective 
    # (since we minimize sum g(k) log(k)/k, positive g(k) increases the sum, i.e., decreases the score).
    # Wait, objective is maximize -sum g(k) log(k)/k.
    # If g(k) = -10, term is 10 log(k)/k (positive contribution to score).
    # If g(k) = 10, term is -10 log(k)/k (negative contribution to score).
    # So we strongly prefer g(k) = -10.
    # The constraint M(x) <= 1 is the bottleneck.
    # M(x) = sum_{k in S_neg} (-10) A(x, k) + sum_{k in S_pos} (10) A(x, k) <= 1.
    # Since A(x, k) < 0, let B(x, k) = -A(x, k) = x/k - floor(x/k) > 0.
    # M(x) = sum_{k in S_neg} 10 B(x, k) - sum_{k in S_pos} 10 B(x, k) <= 1.
    
    # We want to maximize 10 * sum_{k in S_neg} log(k)/k - 10 * sum_{k in S_pos} log(k)/k.
    # Subject to: 10 * (sum_{k in S_neg} B(x, k) - sum_{k in S_pos} B(x, k)) <= 1 for all x.
    # And |f(1)| <= 10. f(1) = -sum g(k)/k.
    # sum g(k)/k = -10 sum_{k in S_neg} 1/k + 10 sum_{k in S_pos} 1/k.
    # f(1) = 10 sum_{k in S_neg} 1/k - 10 sum_{k in S_pos} 1/k.
    # We need |f(1)| <= 10.
    
    # Heuristic:
    # 1. Start with S_neg = empty, S_pos = empty.
    # 2. Iteratively add the k that gives the best "bang for the buck" (increase in score per unit increase in max M(x)).
    #    Actually, we can just try adding k to S_neg if it keeps M(x) <= 1.
    #    Since adding to S_neg increases M(x), we need to check feasibility.
    #    Adding to S_pos decreases M(x), allowing us to add more to S_neg, but costs score.
    #    Likely S_pos is empty or very small. Let's assume S_pos is empty first.
    
    # If S_pos is empty:
    # Constraint: 10 * sum_{k in S_neg} B(x, k) <= 1 => sum_{k in S_neg} B(x, k) <= 0.1 for all x.
    # Score: 10 * sum_{k in S_neg} log(k)/k.
    
    # This is a "knapsack-like" problem where the "weight" of item k is the vector B(., k) 
    # and the capacity is 0.1 for all x.
    # We want to maximize sum value(k) subject to sum weight(k) <= capacity.
    # Greedy approach:
    # Sort k by log(k)/k descending.
    # Try adding them one by one. If adding k violates any constraint, skip it?
    # Or try to add k, and if it violates, maybe we can't add it.
    
    # Let's implement a greedy algorithm that tries to build S_neg.
    
    def check_feasibility(S_neg, K):
        # Returns True if 10 * sum_{k in S_neg} B(x, k) <= 1 for all x in [1, 10K]
        # and |f(1)| <= 10
        if not S_neg:
            return True, 0.0
            
        S_neg = np.array(S_neg)
        # f(1) = 10 * sum(1/k)
        f1 = 10.0 * np.sum(1.0 / S_neg)
        if abs(f1) > 10.0 + 1e-9:
            return False, -np.inf
            
        max_x = 10 * K
        # We only need to check x where B(x, k) might be large.
        # B(x, k) = x/k - floor(x/k) is in [0, 1).
        # The sum is maximized when the fractional parts align.
        # To be safe, check all x? That's expensive.
        # But S_neg is likely small.
        # Let's check x in [1, max_x].
        
        # Vectorized check might be memory heavy if S_neg is large.
        # But S_neg is likely < 100.
        
        # We can compute M(x) for all x.
        # M(x) = 10 * sum_{k in S_neg} (x/k - floor(x/k))
        
        # Optimization: B(x, k) is periodic with period k.
        # The maximum of the sum over x is the maximum over the LCM of the periods?
        # No, the periods are different.
        # We can just sample x. But we need strict guarantee.
        # Let's check all x in [1, max_x].
        # If S_neg has size m, cost is m * 10K.
        # If m=50, 10K=20000 (for K=2000), cost = 1M ops. Fast.
        
        x_vals = np.arange(1, max_x + 1)
        # We can compute the sum for all x by iterating over k in S_neg
        # and adding B(x, k) to a vector.
        
        current_M = np.zeros(max_x)
        for k in S_neg:
            # B(x, k) = x/k - floor(x/k)
            # Vectorized:
            current_M += 10.0 * (x_vals / k - np.floor(x_vals / k))
            
        if np.max(current_M) > 1.0 + 1e-9:
            return False, -np.inf
            
        return True, 10.0 * np.sum(np.log(S_neg) / S_neg)

    # Greedy Search
    # We iterate through candidate K values.
    # For each K, we run the greedy selection.
    
    for K in [500, 1000, 2000, 3000, 5000]:
        if time.time() - start_time > 580:
            break
            
        candidates = np.arange(2, K + 1)
        # Sort by log(k)/k descending
        scores = np.log(candidates) / candidates
        order = np.argsort(scores)[::-1]
        sorted_candidates = candidates[order]
        
        S_neg = []
        current_score = 0.0
        
        # We try to add candidates in order of benefit.
        # If adding a candidate makes the solution infeasible, we skip it.
        # This is a standard greedy heuristic for set packing/covering.
        
        for k in sorted_candidates:
            # Check if adding k to S_neg keeps it feasible
            # We can do this incrementally to avoid re-checking everything from scratch?
            # For now, simple re-check is O(|S_neg| * 10K).
            # With |S_neg| < 100 and K=5000 (10K=50000), it's 5M ops per check.
            # 5000 checks * 5M = 25B ops. Too slow.
            
            # Optimization: 
            # We can maintain the current M(x) vector.
            # When testing a new k, we compute delta_M(x) = 10 * B(x, k).
            # New max M = max(current_M + delta_M).
            # If max <= 1, we accept and update current_M.
            # If not, we reject.
            
            pass # This loop structure needs to be outside the K loop to maintain state?
            # No, S_neg depends on K because the constraint range [1, 10K] changes.
            # However, if a set S is feasible for K, is it feasible for K+1?
            # The constraints are for x in [1, 10K]. For K+1, x in [1, 10(K+1)].
            # The new constraints x in (10K, 10K+10] must also be satisfied.
            # So feasibility is not monotonic in K in a simple way.
            
            # Let's restart the greedy process for each K.
            # To make it faster, we can limit the number of candidates we consider.
            # The top 100 candidates by log(k)/k are the most valuable.
            
            top_candidates = sorted_candidates[:200] # Only consider top 200
            
            S_neg = []
            current_M = np.zeros(10 * K)
            
            for k in top_candidates:
                # Compute delta M for this k
                x_vals = np.arange(1, 10 * K + 1)
                delta_M = 10.0 * (x_vals / k - np.floor(x_vals / k))
                
                new_M = current_M + delta_M
                if np.max(new_M) <= 1.0 + 1e-9:
                    # Check f(1) constraint
                    new_S_neg = S_neg + [k]
                    f1 = 10.0 * np.sum(1.0 / np.array(new_S_neg))
                    if abs(f1) <= 10.0 + 1e-9:
                        S_neg = new_S_neg
                        current_M = new_M
                        current_score += 10.0 * np.log(k) / k
                        
            if S_neg:
                # Final verification
                feasible, score = check_feasibility(S_neg, K)
                if feasible and score > best_score:
                    best_score = score
                    # Construct the final arrays
                    domain = np.arange(1, K + 1)
                    values = np.zeros(K)
                    for k in S_neg:
                        values[k-1] = -10.0
                    # f(1)
                    f1 = 10.0 * np.sum(1.0 / np.array(S_neg))
                    values[0] = f1
                    
                    best_domain = domain
                    best_values = values

    if best_domain is not None:
        return best_domain, best_values
    else:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

if __name__ == "__main__":
    # Quick test
    d, v = prime_number_theorem_partial_function()
    print(f"Best score found: {best_score}")
    print(f"Domain length: {len(d)}")
    print(f"Non-zero values: {np.sum(v != 0)}")

