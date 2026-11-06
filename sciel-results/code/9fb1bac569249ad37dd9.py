import time
from fractions import Fraction
from collections import defaultdict

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit (evaluator kills at 90s, but task says 60s, so 55s is safe)
    
    # Denominators 2 to 48 inclusive
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocals as Fractions for exact arithmetic
    reciprocals = [Fraction(1, d) for d in denominators]
    
    # DP Strategy:
    # We want to find a subset S such that sum(1/d for d in S) < 1.
    # We want to maximize this sum.
    # Let dp be a set of all achievable sums using a subset of the processed denominators.
    # Initially, dp = {Fraction(0)}
    # For each fraction f, new_dp = dp U {s + f for s in dp if s + f < 1}
    # Note: We only care about sums < 1. If s + f >= 1, we discard it because we can't add more 
    # (all terms are positive) and it violates the constraint.
    # Wait, we can't just discard s+f >= 1 immediately if we haven't processed all items?
    # Actually, since all terms are positive, if a partial sum >= 1, any superset will also be >= 1.
    # So we can prune any state >= 1.
    
    # Optimization: 
    # The number of distinct sums might grow large. 
    # Let's monitor the size. If it gets too big, we might need to switch strategies, 
    # but for N=47, the "density" of sums is high, so the number of unique sums < 1 is likely bounded.
    
    # Let's use a set of Fractions.
    # To speed up, we can also keep track of the "best" sum found so far, but we need the full set 
    # to ensure we don't miss a combination that leads to a better sum later.
    # However, if sum A < sum B and both are achievable with the same set of items? No, 
    # items are distinct.
    
    # Is there a dominance relation? 
    # If we have two sums s1 and s2 from the first k items, and s1 < s2, does s2 dominate s1?
    # Not necessarily. s1 might allow adding a large fraction later that s2 cannot.
    # Example: s1=0.1, s2=0.9. Next fraction is 0.95. s1+0.95=1.05 (invalid), s2+0.95=1.85 (invalid).
    # Next fraction is 0.05. s1+0.05=0.15, s2+0.05=0.95. s2 is better.
    # What if next fraction is 0.8? s1+0.8=0.9, s2+0.8=1.7 (invalid). s1 leads to 0.9, s2 leads to 0.9 (if we just keep s2? No, s2 is 0.9, we can't add 0.8).
    # So s1 (0.1) + 0.8 = 0.9. s2 (0.9) is already 0.9.
    # In this case, s2 is at least as good as the result of s1+0.8.
    # Generally, if s2 > s1, any valid extension of s1 (s1 + x < 1) results in s1+x.
    # s2 might not be extendable by x. But s2 itself is already > s1.
    # If s2 cannot be extended by x, its best outcome is s2.
    # If s1 can be extended by x, its best outcome is s1+x.
    # We want max(s2, s1+x).
    # If s1+x > s2, then keeping s1 is useful.
    # If s1+x <= s2, then s2 is better or equal.
    
    # So we cannot simply prune based on magnitude alone. We need to keep the set of achievable sums.
    # Given N=47, let's estimate the number of states.
    # The sum of all reciprocals is ~4.3.
    # The number of distinct sums < 1 is the number of distinct values in {sum(subset) | sum < 1}.
    # This is a subset sum problem over rationals.
    # With denominators up to 48, the LCM is huge, but the number of unique sums is often much smaller than 2^N due to collisions and pruning.
    
    # Let's try the DP. If the set size explodes, we will fall back to the heuristic or just return the best found.
    
    dp = {Fraction(0)}
    
    # We also want to track which set of indices produced the best sum?
    # The task asks for "nonempty distinct zero-based indices".
    # We need to reconstruct the set.
    # Storing the full set for every state in DP is memory heavy.
    # Alternative: Store parent pointers? 
    # Or, since we just need ONE optimal set, we can run the DP to find the optimal value V*, 
    # and then backtrack to find a set that sums to V*.
    
    # However, backtracking in a set-based DP is tricky because multiple subsets can sum to the same value.
    # We can store, for each sum, ONE representative subset (as a bitset or tuple of indices).
    # N=47 fits in a 64-bit integer (or Python int).
    
    # Let's store: sum -> bitmask
    # dp: dict {Fraction: int}
    
    dp = {Fraction(0): 0}
    
    for i in range(n):
        f = reciprocals[i]
        bit = 1 << i
        new_dp = {}
        
        # Start with current states (not taking i)
        for s, mask in dp.items():
            if s not in new_dp:
                new_dp[s] = mask
        
        # Try adding i
        for s, mask in dp.items():
            new_s = s + f
            if new_s < 1:
                # If new_s is already in new_dp, we might want to keep the mask with fewer bits? 
                # Or just any mask? 
                # Any mask is fine as long as it sums to new_s.
                # But we must ensure distinct indices. The mask handles that.
                if new_s not in new_dp:
                    new_dp[new_s] = mask | bit
                else:
                    # We already have a mask for this sum. 
                    # Does it matter which mask we keep?
                    # For future steps, only the sum value matters for feasibility (s < 1).
                    # The mask only matters for reconstruction.
                    # So we can keep the first one we find.
                    pass
        
        dp = new_dp
        
        # Time check
        if time.time() - start_time > timeout:
            break
        
        # Debug/Info: print(len(dp))
        # If len(dp) becomes too large (e.g., > 100,000), we might need to optimize.
        # But for N=47, it should be manageable.
        
    # Find the best sum strictly less than 1
    best_sum = Fraction(0)
    best_mask = 0
    
    for s, mask in dp.items():
        if 0 < s < 1:
            if s > best_sum:
                best_sum = s
                best_mask = mask
    
    # If no valid non-empty set found (should not happen, 1/2 is always valid), fallback
    if best_mask == 0:
        return [0]
        
    # Reconstruct indices from mask
    indices = []
    for i in range(n):
        if best_mask & (1 << i):
            indices.append(i)
            
    return indices

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(res)
    s = sum(Fraction(1, d) for d in [res[i]+2 for i in range(len(res))])
    print(s, 1-s)
