"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
from itertools import combinations
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit

    # The universe is [1, 24]
    universe = list(range(1, 25))
    n = len(universe)

    # Precompute reciprocals as Fractions for exact arithmetic
    recip = {x: Fraction(1, x) for x in universe}

    def is_reciprocal_subsum_free(S):
        """
        Check if set S is reciprocal-subsum-free.
        For every element a in S, 1/a must NOT equal the sum of 1/b for any non-empty subset of S \ {a}.
        """
        S_list = list(S)
        if not S_list:
            return True
        
        for a in S_list:
            others = [x for x in S_list if x != a]
            target = recip[a]
            
            # Check all non-empty subsets of 'others'
            # To optimize, we can prune: if sum of all reciprocals of others < target, no subset can equal target.
            # Also, since all reciprocals are positive, if the minimum reciprocal in others > target, impossible? No, sum can be larger.
            # Actually, since 1/x < 1/a for x > a, and 1/x > 1/a for x < a.
            
            # Efficient check: iterate through subsets.
            # Max size of others is 23. 2^23 is too big.
            # However, we can use a meet-in-the-middle or just simple pruning.
            # Note: The sum of reciprocals of any subset of others.
            
            # Optimization: 
            # 1. If sum(recip[x] for x in others) < target: return False (for this a) immediately? No, if sum < target, then NO subset can equal target. So this 'a' is safe.
            # 2. If any single other x has recip[x] == target: return False. (This implies x == a, but x != a, so impossible for distinct integers).
            # 3. General subset sum.
            
            total_sum = sum((recip[x] for x in others), Fraction(0))
            if total_sum < target:
                continue # Safe for this a
            
            # Check subsets
            # We can sort others by value (larger x -> smaller reciprocal) to help pruning?
            # Let's just do a recursive check or itertools if small.
            # Since n=24, worst case others has 23 elements.
            # But in practice, we are building a set. If the set is large, we might have many elements.
            # However, the property is quite restrictive.
            
            # Let's use a set of achievable sums? No, Fractions are distinct and dense.
            # We'll do a DFS with pruning.
            
            def dfs(idx, current_sum):
                if idx == len(others):
                    return current_sum == target
                # Pruning: 
                # 1. If current_sum == target: 
                #    We need to check if we can reach target with exactly this sum.
                #    Wait, if current_sum == target, we found a subset. Return True.
                #    But we must ensure the subset is non-empty? 
                #    The problem says "sum of reciprocals of a subset". Usually implies non-empty.
                #    If empty subset, sum is 0. 1/a is never 0. So we can ignore empty.
                #    My DFS starts with sum=0. If we return True when current_sum==target, we might return True for empty set if target==0 (impossible).
                #    So if current_sum == target, we have found a valid non-empty subset?
                #    Not necessarily. We might have added nothing yet.
                #    Let's track count or just ensure we don't return True at start.
                #    Actually, since 1/a > 0, and we start at 0, we only hit target after adding at least one element.
                
                if current_sum == target:
                    return True
                
                # Pruning 2: If current_sum > target: return False (all terms positive)
                if current_sum > target:
                    return False
                
                # Pruning 3: If current_sum + sum(others[idx:]) < target: return False
                # Precompute suffix sums for others
                # We'll pass suffix_sum or compute on the fly? Precompute is better.
                
                # Option A: Include others[idx]
                if dfs(idx + 1, current_sum + recip[others[idx]]):
                    return True
                # Option B: Exclude others[idx]
                if dfs(idx + 1, current_sum):
                    return True
                return False

            # Precompute suffix sums for others to enable pruning 3
            # others is a list of integers
            # Let's create a list of their reciprocals
            others_recips = [recip[x] for x in others]
            m = len(others_recips)
            suffix_sums = [Fraction(0)] * (m + 1)
            for i in range(m - 1, -1, -1):
                suffix_sums[i] = suffix_sums[i + 1] + others_recips[i]
            
            # Modify dfs to use suffix_sums
            def dfs_pruned(idx, current_sum):
                if current_sum == target:
                    return True
                if current_sum > target:
                    return False
                if idx == m:
                    return False
                if current_sum + suffix_sums[idx] < target:
                    return False
                
                # Include
                if dfs_pruned(idx + 1, current_sum + others_recips[idx]):
                    return True
                # Exclude
                if dfs_pruned(idx + 1, current_sum):
                    return True
                return False

            if dfs_pruned(0, Fraction(0)):
                return False # Found a subset of others that sums to 1/a
        
        return True

    # Strategy: 
    # We want to maximize the size of S.
    # We can try to build S greedily or use backtracking.
    # Given the constraint is strong, maybe the max size is small?
    # Let's try to find the maximum possible size by checking all subsets of size k from high to low?
    # Or build incrementally.
    
    # Let's try a greedy approach first, then verify/improve with search if time permits.
    # Greedy: Add elements in some order if they don't violate the condition.
    # Which order? 
    # Smaller numbers have larger reciprocals. 
    # 1/1 = 1. Sum of others must be < 1 to be safe? 
    # If we pick 1, we need no subset of others to sum to 1.
    # 1/2 = 0.5.
    # 1/3 = 0.333.
    
    # Let's try to construct a maximal set using a heuristic.
    # Order: Try adding elements from 24 down to 1? Or 1 to 24?
    # Large numbers have small reciprocals. It's easier to avoid summing to a specific value with small numbers?
    # Actually, if we pick large numbers, their reciprocals are small. 
    # 1/24 = 0.0416.
    # Sum of many small reciprocals can equal a larger reciprocal.
    # Example: 1/6 = 1/12 + 1/12 (not allowed, distinct). 1/6 = 1/12 + 1/24 + 1/24? No distinct.
    # 1/2 = 1/3 + 1/6. If we have {2, 3, 6}, then 1/2 = 1/3 + 1/6. So {2,3,6} is invalid.
    
    # Let's run a search for the maximum cardinality.
    # We can iterate k from 24 down to 1 and check if any subset of size k is valid.
    # This is expensive. 
    # Better: Branch and Bound.
    
    best_set = []
    
    def backtrack(current_set, candidates):
        nonlocal best_set
        if time.time() > deadline:
            return
        
        if len(current_set) > len(best_set):
            best_set = list(current_set)
        
        if not candidates:
            return
            
        for i in range(len(candidates)):
            x = candidates[i]
            new_set = current_set + [x]
            if is_reciprocal_subsum_free(new_set):
                backtrack(new_set, candidates[i+1:])
            # Optional: if time is running out, stop exploring
            if time.time() > deadline:
                return

    # Initial candidates: all elements
    # To find a large set, we should probably try to include many elements.
    # Let's sort candidates. 
    # Does order matter for finding a large set? 
    # If we put larger numbers first, we might fill the set with "safe" small reciprocals?
    # Or smaller numbers first?
    # 1 is very restrictive. 1/1 = 1. No subset of others can sum to 1.
    # Sum of all reciprocals of [2..24] is H_24 - 1 ≈ 3.77 - 1 = 2.77. So 1 is definitely representable?
    # 1 = 1/2 + 1/3 + 1/7 + 1/42 ... 
    # If 1 is in the set, it's very hard to satisfy.
    # So maybe exclude 1?
    
    # Let's try multiple orders or just a good heuristic.
    # Heuristic: Start with large numbers, try to add smaller ones.
    # Candidates sorted descending: 24, 23, ..., 1
    
    candidates_desc = sorted(universe, reverse=True)
    backtrack([], candidates_desc)
    
    # If best_set is empty or small, try ascending?
    if len(best_set) < 5:
        candidates_asc = sorted(universe)
        backtrack([], candidates_asc)
        
    # Ensure we return a non-empty set
    if not best_set:
        return [24] # Fallback
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
