"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    
    # Greedy construction with subset-sum DP verification
    # Key insight: An element a is the mean of subset S of others iff
    # sum(S) = len(S) * a. We need to check for each a in selected,
    # whether there exists a subset of OTHER elements with sum = k*a for some k >= 2.
    
    selected = []
    n_max = 512
    
    def can_add(a, current_set):
        """Check if adding a to current_set violates the condition."""
        # For each element x in current_set ∪ {a}, check if x is mean of any subset of others
        # We only need to check:
        # 1. a itself: is a the mean of any subset of current_set (size >= 2)?
        # 2. Each x in current_set: is x the mean of any subset of (current_set - {x}) ∪ {a} (size >= 2)?
        
        # Check 1: Is a the mean of any subset of current_set with size >= 2?
        # This means sum(subset) = k * a for some k >= 2, subset ⊆ current_set
        # Use subset sum DP: for each k, check if there's a subset of current_set with sum = k*a
        
        # Check 2: For each x in current_set, is x the mean of a subset of (current_set - {x}) ∪ {a} with size >= 2?
        # This means sum(subset) = k * x for some k >= 2, subset ⊆ (current_set - {x}) ∪ {a}
        
        # Optimization: Since we're building greedily, we can use DP to track achievable sums.
        # For a given set S, compute which sums are achievable with subsets of size k.
        
        # For check 1: subsets of current_set
        # For check 2: subsets of (current_set - {x}) ∪ {a}
        
        # Let's use a bitset approach for subset sums by cardinality
        # dp[k] = set of achievable sums using exactly k elements
        
        def check_mean(target, available):
            """Check if target is the mean of any subset of available with size >= 2."""
            # Need: exists subset T ⊆ available, |T| >= 2, sum(T) = |T| * target
            # i.e., sum(T) - |T| * target = 0
            # Let v_i = element_i - target for each element in available
            # Need: exists subset T, |T| >= 2, sum_{i in T} v_i = 0
            
            # Use DP: dp[k] = set of achievable sums of (element - target) using k elements
            # Check if 0 is in dp[k] for any k >= 2
            
            n = len(available)
            if n < 2:
                return False
            
            # dp[k] is a set of achievable sums
            # To save memory, use bitsets or sets
            # Since values can be negative, use offset
            
            # For efficiency, use a set-based DP
            # dp[k] = set of sums achievable with exactly k elements from available
            dp = [set() for _ in range(n + 1)]
            dp[0] = {0}
            
            for elem in available:
                v = elem - target
                for k in range(n, 0, -1):
                    if dp[k-1]:
                        for s in dp[k-1]:
                            dp[k].add(s + v)
            
            for k in range(2, n + 1):
                if 0 in dp[k]:
                    return True
            return False
        
        # Check 1: a is mean of subset of current_set (size >= 2)
        if len(current_set) >= 2:
            if check_mean(a, current_set):
                return False
        
        # Check 2: For each x in current_set, x is mean of subset of (current_set - {x}) ∪ {a} (size >= 2)
        for x in current_set:
            others = [y for y in current_set if y != x] + [a]
            if len(others) >= 2:
                if check_mean(x, others):
                    return False
        
        return True
    
    # Greedy: try to add numbers from 1 to 512
    for a in range(1, n_max + 1):
        if time.time() > deadline:
            break
        if can_add(a, selected):
            selected.append(a)
            if len(selected) >= 48:
                break
    
    # Try to improve with random restarts if time permits
    if len(selected) < 48 and time.time() < deadline - 5:
        best = selected[:]
        
        for restart in range(50):
            if time.time() > deadline - 3:
                break
            
            # Randomized greedy: shuffle the order
            candidates = list(range(1, n_max + 1))
            random.shuffle(candidates)
            
            current = []
            for a in candidates:
                if time.time() > deadline - 3:
                    break
                if can_add(a, current):
                    current.append(a)
                    if len(current) >= len(best):
                        if len(current) > len(best):
                            best = current[:]
                        break
            
            if len(current) > len(best):
                best = current[:]
    
    return best[:48]
# EVOLVE-BLOCK-END
