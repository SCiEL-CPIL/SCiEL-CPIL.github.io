"""Rewritten program: structural change to a bitset-based constraint solver with randomized local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill

    N = 512
    MAX_K = 48

    # Precompute for each target sum S and size k, the list of elements that are the mean
    # i.e., for a candidate set S, element a is bad if there exists a subset T of S\{a}
    # of size k>=2 such that sum(T) = k*a.
    # Equivalent: for each a in S, and each k>=2, is there a subset of S\{a} of size k with sum k*a?
    # We use bitset DP: dp[s][k] = bitset of elements achievable as sum of k elements.
    # But we need to check for each a, whether a is in dp[k*a][k] when we remove a.
    # This is expensive to do per-a. Instead, we can compute dp_all[s][k] for the full set,
    # and then check if removing a still allows the sum. But that's also complex.
    
    # Alternative: for a given set, check validity by:
    # For each a in set:
    #   For each k in 2..len(set)-1:
    #     Check if there's a subset of size k from set\{a} with sum k*a.
    # Use bitset DP on set\{a}.
    
    # To speed up: precompute subset-sum bitsets for the full set, then for each a,
    # we need to know if k*a can be formed by k elements NOT including a.
    
    # Let's define a helper that checks validity of a set using bitset DP.
    
    def is_valid(set_vals):
        """Check if no element is the mean of any subset of >=2 other elements."""
        m = len(set_vals)
        if m < 3:
            return True
        vals = sorted(set_vals)
        # dp[s][k] = bitset of which sums of size k are achievable? 
        # Actually, we want: for each k and target sum T, can we form T with k elements?
        # Use: dp[k][s] = 1 if we can form sum s with k elements.
        # But we need to exclude specific element a.
        
        # Precompute dp_all[k][s] for the full set.
        # Then for element a, we need dp_excl_a[k][s] which is dp_all minus contributions involving a.
        # This is hard. Instead, for each a, do a DP on set\{a}.
        
        # For efficiency, let's precompute for each a the DP on set\{a}.
        # But that's O(n^3) which is too slow for n=48.
        
        # Alternative: use the full set DP, and for each a, check if k*a is achievable
        # by k elements where at least one is not a... no, we need ALL k elements to be != a.
        
        # Let's just do: for each a, build a bitset DP on the other elements.
        # n=48, so for each a, we do DP on 47 elements.
        # DP: dp[k][s] = bitset? No, just boolean.
        # Actually, we can use: for each a, we want to know for each k>=2, is k*a in the subset sums of size k from others?
        
        # Let's precompute for the full set: subset_sum[k][s] = True if we can form sum s with k elements.
        # Then for element a, we need to check if there's a way to form k*a with k elements NOT including a.
        # This is equivalent to: in the full set, can we form k*a with k elements, AND is there such a representation not using a?
        # If the only way to form k*a with k elements uses a, then it's fine for a.
        
        # This is getting complex. Let's just do the straightforward check for small sets.
        
        # For n<=48, we can do: for each a, for each k in 2..n-1, check if k*a is a subset sum of size k from others.
        # Use bitset DP: dp[s] = bitmask of sizes achievable? No.
        # dp[k] = set of sums achievable with k elements.
        
        # Let's use: for a given set of values, compute all subset sums by size.
        # dp[k] = set of sums achievable with exactly k elements.
        
        for a in vals:
            others = [v for v in vals if v != a]
            n_others = len(others)
            # dp[k] = set of sums achievable with k elements from others
            dp = [set() for _ in range(n_others + 1)]
            dp[0].add(0)
            for v in others:
                for k in range(n_others, 0, -1):
                    dp[k] = dp[k] | {s + v for s in dp[k-1]}
            for k in range(2, n_others + 1):
                if k * a in dp[k]:
                    return False
        return True

    # This is_valid is O(n^4) which is too slow for n=48 in Python.
    # Let's optimize with bitsets.
    
    def is_valid_fast(set_vals):
        """Faster validity check using bitset DP."""
        m = len(set_vals)
        if m < 3:
            return True
        vals = sorted(set_vals)
        
        # For each element a, check if k*a is achievable as sum of k other elements.
        # Use bitset: dp[s] = bitmask of counts? No.
        # Let's use: for a fixed set of others, we want to know for each k, what sums are achievable.
        # We can use: dp[k] as a bitset of sums, but sums can be up to 47*512 ~ 24064, which is fine.
        
        # Actually, let's precompute for the full set, then for each a, adjust.
        # Simpler: just do the check per a with a fast DP.
        
        for a in vals:
            others = [v for v in vals if v != a]
            n_others = len(others)
            # Max sum: n_others * 512
            max_sum = n_others * 512
            # dp[k] = bitset of sums achievable with k elements
            # Use Python integers as bitsets.
            dp = [0] * (n_others + 1)
            dp[0] = 1  # sum 0 achievable with 0 elements
            for v in others:
                for k in range(n_others, 0, -1):
                    # dp[k] |= dp[k-1] << v
                    dp[k] |= dp[k-1] << v
            for k in range(2, n_others + 1):
                target = k * a
                if target <= max_sum and (dp[k] >> target) & 1:
                    return False
        return True

    # Now, let's use a local search approach.
    # Start with a random set of size 48, then iteratively improve.
    
    # First, let's try a greedy construction to get a good initial set.
    def greedy_construct():
        selected = []
        for x in range(1, N + 1):
            candidate = selected + [x]
            if is_valid_fast(candidate):
                selected = candidate
                if len(selected) >= MAX_K:
                    break
        return selected

    # Start with greedy
    current_set = greedy_construct()
    current_size = len(current_set)
    best_set = current_set[:]
    best_size = current_size

    print(f"Greedy found {current_size} elements")

    # Now try local search to see if we can find a larger set or improve.
    # Actually, the task is to maximize cardinality, so if greedy found 48, we're done.
    # Let's check.
    if current_size >= MAX_K:
        return current_set

    # If greedy didn't find 48, try local search to find a better set.
    # Represent the set as a boolean array of size N+1.
    # Use simulated annealing or simple hill climbing.
    
    # For now, let's just return the greedy result if we're running low on time.
    if time.time() > deadline - 10:
        return best_set

    # Try to improve by swapping elements.
    random.seed(42)
    current = set(current_set)
    
    for iteration in range(10000):
        if time.time() > deadline - 5:
            break
        # Try to add a random element not in current
        if len(current) < MAX_K:
            candidates = [x for x in range(1, N + 1) if x not in current]
            if candidates:
                x = random.choice(candidates)
                new_set = current | {x}
                if is_valid_fast(new_set):
                    current = new_set
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = list(current)
                        print(f"Improved to {best_size} at iteration {iteration}")
        # Try to swap
        if current:
            a = random.choice(list(current))
            candidates = [x for x in range(1, N + 1) if x not in current]
            if candidates:
                b = random.choice(candidates)
                new_set = (current - {a}) | {b}
                if is_valid_fast(new_set):
                    current = new_set
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = list(current)
                        print(f"Improved to {best_size} at iteration {iteration}")

    return best_set
# EVOLVE-BLOCK-END
