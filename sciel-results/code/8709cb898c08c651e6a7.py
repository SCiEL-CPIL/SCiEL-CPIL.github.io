"""
Exact search for maximum-cardinality dissociated reciprocal subset of [1..22].
Uses LCM normalization and iterative deepening to avoid premature pruning.
"""
import time
from math import lcm
from functools import reduce

def construct():
    start = time.time()
    deadline = start + 82.0

    L = reduce(lcm, range(1, 23))
    N = 22
    vals = [L // i for i in range(1, N + 1)]

    # Precompute for each index i, the sorted list of values after i
    # and a bit-mask of all elements (for quick "can't beat" check)
    
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]  # fallback: 9 elements
    best_size = len(best)

    def time_check():
        return time.time() > deadline

    # For each target size from 22 down to best_size + 1,
    # try to find a dissociated subset of that size.
    # If found, update best and stop (since we go descending).
    
    # However, descending from 22 is very slow because 22 is impossible.
    # Let's estimate: max possible size.
    # The sum of reciprocals of [1..22] is about 3.69.
    # A dissociated set requires 2^k distinct subset sums.
    # The total sum is S = sum(vals). The sums are in [0, S].
    # So 2^k <= S + 1.
    # S = L * sum(1/i for i=1..22).
    # L is huge, so this bound is weak.
    
    # Better bound:
    # The smallest element in the set is at least 1.
    # If the set contains 1, then 1 is the smallest increment.
    # The number of distinct sums is 2^k.
    # They must fit in the range [0, sum].
    # But more importantly, if we sort the selected elements a_1 < a_2 < ... < a_k,
    # and scale by L, we get v_1 > v_2 > ... > v_k.
    # The smallest difference between two distinct subset sums is at least min(v_k, v_k/2, ...)?
    # Actually, if the set is dissociated, all 2^k sums are distinct.
    # The smallest positive difference between two subset sums is at least the smallest "atom" that can form a difference.
    # This is related to the concept of "dissociated" in additive combinatorics.
    
    # Let's just do a depth-first search that tries to build a set of size K.
    # We can use the bound: if current_size + (N - idx) < K, prune.
    
    def exists_dissociated_of_size(K):
        nonlocal best, best_size, deadline, time_check
        
        if K <= best_size:
            return False
            
        # Try to find any set of size K
        # Use DFS with index
        # State: idx (next index to consider), current_sums (frozenset or set), current_subset (list), current_count
        
        # To speed up, we can order indices.
        # Indices with larger values (smaller i) might be "safer" to include?
        # Or indices with smaller values (larger i) create finer granularity?
        # Usually, including larger values (smaller i) makes sums larger, potentially avoiding collisions?
        # But smaller values (larger i) are more likely to cause collisions if they are small differences.
        # Let's try sorting indices by value descending (i.e., i ascending) first.
        # Or maybe try to include "large" values first to push sums apart.
        
        # Let's try a heuristic: sort indices by value descending.
        # Indices: 1 (val=L), 2 (val=L/2), ..., 22 (val=L/22).
        # So i=1 is largest, i=22 is smallest.
        
        # We'll iterate i from 0 to N-1 (corresponding to 1..22).
        # But we can permute the order of trying indices.
        # Let's try the natural order first, but with strong pruning.
        
        def dfs(idx, current_sums, current_subset, current_count):
            nonlocal best, best_size, time_check
            
            if time_check():
                return False
                
            if current_count == K:
                # Found a valid set of size K
                best = current_subset[:]
                best_size = K
                return True
                
            # Pruning: if we can't reach size K
            if current_count + (N - idx) < K:
                return False
                
            for i in range(idx, N):
                if time_check():
                    return False
                    
                v = vals[i]
                
                # Check if adding v creates a collision
                # new_sums = {s + v for s in current_sums}
                # Collision if new_sums intersects current_sums
                
                # Optimization: if v is very small, it might collide easily.
                # Let's just do the check.
                
                valid = True
                new_sums = set()
                for s in current_sums:
                    ns = s + v
                    if ns in current_sums:
                        valid = False
                        break
                    new_sums.add(ns)
                    
                if valid:
                    # Also need to ensure new_sums doesn't have internal collisions?
                    # s1 + v == s2 + v => s1 == s2. Since current_sums is a set, s1 != s2 implies s1+v != s2+v.
                    # So no internal collisions in new_sums.
                    # We just need disjointness from current_sums.
                    
                    # Optimization: if the set is large, this check is O(2^current_count).
                    # For K=16, 2^16 = 65536. This might be slow if we do it many times.
                    # But we prune heavily.
                    
                    new_total_sums = current_sums | new_sums
                    
                    current_subset.append(i + 1)
                    if dfs(i + 1, new_total_sums, current_subset, current_count + 1):
                        return True
                    current_subset.pop()
                    
        return dfs(0, {0}, [], 0)

    # Try sizes from 22 down to best_size + 1
    # Start from a reasonable upper bound.
    # Let's try 16 first (current best is 16? No, current best is 9 in fallback, but search found 16).
    # Wait, the current program score is 1.7778 which is 16/9.
    # So the current program already finds 16.
    # Can we find 17?
    
    # Let's try 17, 18, ... up to 22.
    # If we find 17, we return it.
    
    # To avoid wasting time on impossible sizes, we can start from a smaller K if we suspect the max is low.
    # But 16 is already found. Let's try 17.
    
    # Optimization: If we fail to find K, we don't know if K-1 is possible, but we have best_size from previous runs or fallback.
    # Actually, we should start from the highest possible and go down.
    
    # However, searching for K=22 is hopeless.
    # Let's use a quick upper bound.
    # The sum of reciprocals of [1..n] is H_n.
    # For a dissociated set, the number of distinct sums is 2^k.
    # The sums are multiples of 1/L.
    # The largest sum is sum(vals) = L * H_22.
    # So 2^k <= L * H_22 + 1.
    # This is always true for small k.
    
    # A better bound:
    # If the set is dissociated, the elements are "linearly independent" over {0,1} in the group Z.
    # This implies that no element can be written as a sum of differences of other subset sums.
    
    # Let's just try K=17. If it fails within time, try K=16 (already known to work, so skip if best_size >= 16).
    # Since best_size starts at 9, and we know 16 works, we can set best_size=16 initially if we are confident.
    # But to be safe, let's run the search for K=17.
    
    # Actually, the previous program found 16.
    # If we set best_size=16, we can just try K=17, 18, etc.
    # But we don't know for sure if 16 is the max.
    
    # Let's try to find the max.
    # Start K from 22 down to 1.
    # But add a time limit per K.
    
    # Revised plan:
    # 1. Set a global time limit.
    # 2. Iterate K from 22 down to 1.
    # 3. For each K, run DFS with a time budget (e.g., 5 seconds).
    # 4. If found, return.
    # 5. If time runs out, return best found so far.
    
    # But DFS for K=22 will take forever.
    # We need a better upper bound.
    
    # Let's use the fact that if we include 1, then 1 is in the set.
    # If we include 1 and 2, then 1, 2, 1+2=3 are sums.
    # If we include 3, then 3 is a sum, which collides with 1+2.
    # So {1, 2, 3} is not dissociated.
    
    # General rule: if a, b, c are in the set and a+b=c, then {a,b,c} is not dissociated.
    # More generally, if a subset sum equals another subset sum, it's not dissociated.
    
    # Let's try a greedy approach to get a lower bound, then search for higher.
    
    # Greedy:
    # Start with empty set.
    # For i from 1 to 22:
    #   If adding i keeps the set dissociated, add it.
    
    def greedy_build():
        current_sums = {0}
        current_subset = []
        for i in range(1, N + 1):
            v = vals[i - 1]
            valid = True
            new_sums = set()
            for s in current_sums:
                ns = s + v
                if ns in current_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if valid:
                current_sums = current_sums | new_sums
                current_subset.append(i)
        return current_subset

    # Run greedy to get a starting point
    greedy_result = greedy_build()
    if len(greedy_result) > best_size:
        best = greedy_result
        best_size = len(best)
    
    # Now try to improve beyond best_size
    # Start from best_size + 1 up to N
    
    # Actually, we want the MAXIMUM.
    # So we should try from N down to best_size + 1.
    
    # But N=22 is impossible.
    # Let's try 17, 18, ...
    
    # To save time, we can skip K values that are obviously too large.
    # But we don't have a good upper bound.
    
    # Let's just try K = best_size + 1, best_size + 2, ...
    # And if we find one, we update best and try to find larger ones.
    # But this might miss the maximum if it's not contiguous.
    # However, if there is a dissociated set of size K, there is one of size K-1 (just remove one element).
    # So the set of possible sizes is contiguous from 1 to max_size.
    # Therefore, we can just incrementally increase K.
    
    # So:
    # K = best_size + 1
    # while K <= N and not time_check():
    #   if exists_dissociated_of_size(K):
    #       best_size = K
    #       K += 1
    #   else:
    #       break  # Since sizes are contiguous, if K fails, K+1 fails too.
    
    # This is efficient!
    
    K = best_size + 1
    while K <= N:
        if time_check():
            break
        if exists_dissociated_of_size(K):
            K += 1
        else:
            break
    
    return best

# EVOLVE-BLOCK-END
