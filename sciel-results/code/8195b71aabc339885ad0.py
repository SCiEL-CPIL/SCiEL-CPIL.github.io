import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precompute reciprocals
    recip = {i: Fraction(1, i) for i in range(1, 25)}
    candidates = list(range(1, 25))
    n = len(candidates)
    
    # Build conflict graph: conflict[i][j] = True if i and j cannot both be selected
    # Two elements conflict if:
    # 1. 1/i = 1/j (impossible for distinct i,j)
    # 2. 1/i is a subset sum of reciprocals of other selected elements
    # For pairwise: 1/i = 1/j is never true for distinct i,j in [1,24]
    # But we need to check: can 1/i be written as sum of a subset of {1/j : j in S, j != i}?
    # This is NOT just pairwise - it depends on the full set S.
    
    # However, we can define a stronger conflict: i conflicts with j if 1/i = 1/j (never)
    # OR if {i, j} alone creates a violation. But violations can involve 3+ elements.
    
    # Alternative approach: Use branch and bound with incremental validity checking.
    # Precompute for each element x, which subsets of OTHER elements sum to 1/x.
    
    # For each x, find all subsets T of [1,24]\{x} such that sum(1/t for t in T) = 1/x
    # Then x cannot be in any valid set that contains all of T.
    
    # Let's precompute: for each x, the set of "forbidden subsets" - subsets T of [1,24]\{x} 
    # where sum of reciprocals equals 1/x. If S contains x AND S contains all of T, S is invalid.
    
    # This is still complex. Let's use a different approach:
    # Branch and bound where we maintain the current set and check validity incrementally.
    
    # Key optimization: when adding element x to current set C, we only need to check:
    # 1. Does 1/x equal sum of reciprocals of some subset of C? (x is "covered")
    # 2. For each y in C, does 1/y now equal sum of reciprocals of some subset of C∪{x} \ {y}?
    #    This only changes if the new subset includes x. So check: does 1/y - 1/x = sum of reciprocals of some subset of C\{y,x}?
    
    best_set = []
    best_size = 0
    
    def can_form_target(target, available, memo=None):
        """Check if target can be formed as sum of a subset of available reciprocals."""
        if target == 0:
            return True
        if not available or sum(available) < target:
            return False
        if memo is None:
            memo = {}
        key = (target, tuple(sorted(available, key=lambda f: (f.numerator, f.denominator))))
        if key in memo:
            return memo[key]
        
        # Sort in descending order for better pruning
        avail_sorted = sorted(available, reverse=True)
        
        def dfs(idx, current):
            if current == target:
                return True
            if idx >= len(avail_sorted) or current > target:
                return False
            # Prune: even adding all remaining can't reach target
            if current + sum(avail_sorted[idx:]) < target:
                return False
            
            # Include avail_sorted[idx]
            if dfs(idx + 1, current + avail_sorted[idx]):
                return True
            # Exclude
            if dfs(idx + 1, current):
                return True
            return False
        
        result = dfs(0, Fraction(0))
        memo[key] = result
        return result
    
    # Memoization for subset sum checks
    subset_sum_memo = {}
    
    def subset_sum_check(target, element_set):
        """Check if target can be formed from subset of reciprocals of elements in element_set."""
        if target == 0:
            return True
        if not element_set:
            return False
        
        # Use frozenset as key
        key = (target, frozenset(element_set))
        if key in subset_sum_memo:
            return subset_sum_memo[key]
        
        avail = [recip[e] for e in element_set]
        if sum(avail) < target:
            subset_sum_memo[key] = False
            return False
        
        avail_sorted = sorted(avail, reverse=True)
        
        def dfs(idx, current):
            if current == target:
                return True
            if idx >= len(avail_sorted) or current > target:
                return False
            if current + sum(avail_sorted[idx:]) < target:
                return False
            
            if dfs(idx + 1, current + avail_sorted[idx]):
                return True
            if dfs(idx + 1, current):
                return True
            return False
        
        result = dfs(0, Fraction(0))
        subset_sum_memo[key] = result
        return result
    
    def is_valid_set(S):
        """Check if set S is valid (no reciprocal subsums)."""
        S_list = list(S)
        for x in S_list:
            others = [y for y in S_list if y != x]
            if subset_sum_check(recip[x], others):
                return False
        return True
    
    # Branch and bound
    def branch_and_bound(remaining, current_set, depth):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        # Upper bound: current size + number of remaining elements
        upper_bound = len(current_set) + len(remaining)
        if upper_bound <= best_size:
            return
        
        if not remaining:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            return
        
        # Try each element in remaining
        for i, x in enumerate(remaining):
            if time.time() > deadline:
                return
            
            # Check if adding x to current_set is valid
            new_set = current_set + [x]
            if is_valid_set(new_set):
                # Recurse with remaining elements after i
                remaining_after = remaining[i+1:]
                branch_and_bound(remaining_after, new_set, depth + 1)
            
            # Also try excluding x (already handled by continuing loop)
    
    # Initial greedy to get a good lower bound
    greedy_set = []
    for x in sorted(candidates, reverse=True):  # Start with larger elements (smaller reciprocals)
        test_set = greedy_set + [x]
        if is_valid_set(test_set):
            greedy_set = test_set
    
    best_set = greedy_set[:]
    best_size = len(greedy_set)
    
    # Now run branch and bound from the start
    # To make it more effective, try different orderings
    for order in [sorted(candidates, reverse=True), sorted(candidates), sorted(candidates, key=lambda x: (x % 2, -x))]:
        if time.time() > deadline:
            break
        branch_and_bound(order[:], [], 0)
    
    # If we still have time, try local search / repair
    if time.time() < deadline - 5:
        # Try to improve best_set by swapping elements
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            current_best = best_set[:]
            for i, x in enumerate(current_best):
                if time.time() > deadline - 2:
                    break
                # Try removing x and adding each element not in current_best
                for y in candidates:
                    if y in current_best:
                        continue
                    test_set = current_best[:i] + current_best[i+1:] + [y]
                    if is_valid_set(test_set) and len(test_set) > len(best_set):
                        best_set = test_set
                        best_size = len(test_set)
                        improved = True
                        break
    
    return best_set if best_set else [13]
