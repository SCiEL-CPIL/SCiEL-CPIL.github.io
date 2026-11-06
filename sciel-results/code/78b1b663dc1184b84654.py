"""Improved program: greedy with fast feasibility check using subset-sum DP."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    def is_feasible(selected, candidate):
        """Check if adding candidate to selected is feasible.
        An element a is bad if it's the mean of some subset of >=2 OTHER elements.
        For each element a in selected+[candidate], check if sum(subset) == len(subset)*a
        for some subset of other elements of size >= 2.
        
        We use subset-sum DP: for a given target T and max count, check if there's a subset
        of size k>=2 with sum T from the other elements.
        """
        elements = selected + [candidate]
        n = len(elements)
        
        for i, a in enumerate(elements):
            others = [x for j, x in enumerate(elements) if j != i]
            # Need subset of others, size >= 2, sum = k*a for some k >= 2
            # Max possible sum is sum(others), so k*a <= sum(others) => k <= sum(others)//a
            max_k = sum(others) // a
            if max_k < 2:
                continue
            # Check for each k from 2 to max_k if there's a subset of size k with sum k*a
            # Use DP: dp[s][c] = True if we can pick c elements summing to s
            # But this is expensive. Let's use a different approach.
            # 
            # Alternative: For each k in 2..max_k, check if there's a subset of size k with sum k*a.
            # This is a bounded subset-sum problem.
            # 
            # Actually, let's think differently. The condition is:
            # a = (sum of subset) / len(subset)
            # So sum(subset) = a * len(subset)
            # 
            # Let's use DP over the other elements, tracking (sum, count).
            # But sums can be up to 512*48 = 24576, counts up to 48.
            # That's 24576 * 48 ~ 1.2M states, which is manageable.
            
            max_sum = sum(others)
            # dp[c] = set of sums achievable with c elements
            # Or better: dp[s] = set of counts, or use a 2D boolean array
            # Let's use: reachable[s][c] but that's large. 
            # Instead, for each k, we can do a subset-sum check.
            
            # More efficient: for each k from 2 to min(max_k, len(others)):
            #   target = k * a
            #   Check if there's a subset of others of size k with sum = target
            #   This can be done with DP: dp[j][s] = can pick j elements summing to s
            #   But doing this for each k is expensive.
            
            # Better approach: single DP pass
            # dp[c][s] = True if we can select c elements from processed so far summing to s
            # We iterate over others, updating dp.
            
            max_count = len(others)
            # dp[c][s] - use a list of sets or bitsets
            # Let's use: for each count c, a set of achievable sums
            # Or use numpy bitsets for speed
            
            # dp[c] is a set of sums
            dp = [set() for _ in range(max_count + 1)]
            dp[0].add(0)
            
            for x in others:
                # Update in reverse to avoid using x multiple times
                for c in range(max_count, 0, -1):
                    for s in list(dp[c-1]):
                        new_s = s + x
                        if new_s <= max_sum:
                            dp[c].add(new_s)
            
            # Now check: for any k >= 2, is k*a in dp[k]?
            for k in range(2, max_count + 1):
                if k * a in dp[k]:
                    return False
        
        return True
    
    # Greedy approach: try to build a large set
    # Start with a good initial set, then try to add more elements
    
    # First, let's try a simple greedy: go through 1..512 and add if feasible
    selected = []
    for x in range(1, 513):
        if time.time() - start_time > DEADLINE:
            break
        if is_feasible(selected, x):
            selected.append(x)
        if len(selected) >= 48:
            break
    
    # Try to improve with random restarts
    best = selected[:]
    
    if time.time() - start_time < DEADLINE - 5:
        # Try different orderings
        rng = random.Random(42)
        for trial in range(50):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Random permutation of 1..512
            candidates = list(range(1, 513))
            rng.shuffle(candidates)
            
            selected2 = []
            for x in candidates:
                if time.time() - start_time > DEADLINE - 5:
                    break
                if is_feasible(selected2, x):
                    selected2.append(x)
                if len(selected2) >= 48:
                    break
            
            if len(selected2) > len(best):
                best = selected2
                if len(best) >= 48:
                    break
    
    return best[:48]
# EVOLVE-BLOCK-END
