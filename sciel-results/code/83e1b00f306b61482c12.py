import time
from fractions import Fraction
from math import lcm

_START_TIME = time.time()
_DEADLINE = 85.0

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

# Precompute LCM of 1..24
_L = 1
for i in range(1, 25):
    _L = lcm(_L, i)

# Precompute integer reciprocals: R[x] = _L // x
_R = [0] * 25
for i in range(1, 25):
    _R[i] = _L // i

def _subset_sum_exists(target, elements):
    """
    Check if target can be formed by a non-empty subset of elements.
    elements is a list of integers (scaled reciprocals).
    target is an integer (scaled reciprocal).
    """
    if not elements:
        return False
    if target < 0:
        return False
    
    # If target is 0, we need a non-empty subset summing to 0.
    # But all elements are positive, so sum > 0.
    # So if target == 0, return False (unless we allow empty set, but we don't).
    # Wait, the problem says "subset of the OTHER selected elements".
    # "No reciprocal ... may equal the sum of reciprocals of a subset".
    # Usually "subset" can be empty? "sum of reciprocals of a subset".
    # If the subset is empty, sum is 0. 1/k is never 0. So empty set is irrelevant.
    
    # Optimization: if sum(elements) < target, return False.
    # if min(elements) > target, return False (if we need at least one element).
    # Actually, we can just do DFS or set-based DP.
    
    # Set-based DP is O(2^n) in worst case, but often prunes.
    # Given n <= 22, 2^22 is 4M. Might be slow if called many times.
    # But in backtracking, n is small.
    
    # Let's use a set of reachable sums.
    reachable = {0}
    for e in elements:
        # We can also prune: if s + e > target, we don't need to add it to reachable
        # if we only care about exact match?
        # No, we need to know if target is reachable.
        # We can filter reachable to only keep sums <= target.
        new_reachable = set()
        for s in reachable:
            if s <= target:
                new_reachable.add(s)
            if s + e <= target:
                new_reachable.add(s + e)
        reachable = new_reachable
        
    return target in reachable

def _is_valid(S):
    """
    Check if S is a valid reciprocal-subsum-free set.
    S is a list of integers.
    """
    for k in S:
        # Check 1/k
        target = _R[k]
        others = [x for x in S if x != k]
        elems = [_R[x] for x in others]
        if _subset_sum_exists(target, elems):
            return False
            
    return True

def construct():
    # Try to find the largest valid set
    # We know 12 is valid. Greedy found ~19.
    # Let's try 24 down to 1.
    
    # To speed up, we can use a greedy heuristic to get a lower bound,
    # then search for larger sets.
    
    # Greedy descending
    S_greedy = []
    for k in range(24, 0, -1):
        if _is_valid(S_greedy + [k]):
            S_greedy.append(k)
            
    max_size = len(S_greedy)
    best_set = S_greedy
    
    # Try to find a set of size max_size + 1, max_size + 2, etc.
    # We can use backtracking.
    
    # Optimization: Sort elements in a specific order for backtracking.
    # Usually, small numbers are more restrictive.
    # Let's try 1 to 24.
    
    def backtrack(idx, current_set, current_size):
        nonlocal max_size, best_set
        
        if _time_remaining() < 2:
            return False
            
        # Pruning: if we can't reach a larger size, stop
        # We are searching for a set of size > max_size.
        # If current_size + (24 - idx) <= max_size, we can't beat max_size.
        if current_size + (24 - idx) <= max_size:
            return False
            
        if idx == 24:
            if current_size > max_size:
                max_size = current_size
                best_set = list(current_set)
            return current_size > max_size
            
        # Option 1: Exclude idx+1
        if backtrack(idx + 1, current_set, current_size):
            return True
            
        # Option 2: Include idx+1
        k = idx + 1
        new_set = current_set + [k]
        
        # Quick check: is new_set valid?
        # We only need to check if adding k broke the validity of existing elements
        # or if k itself is a subsum.
        
        # 1. Check if 1/k is a subsum of current_set
        target_k = _R[k]
        elems_k = [_R[x] for x in current_set]
        if _subset_sum_exists(target_k, elems_k):
            return backtrack(idx + 1, current_set, current_size) # Exclude k
        
        # 2. Check if adding k creates a subsum for any existing y
        valid = True
        for y in current_set:
            target_y = _R[y]
            # We need to check if 1/y can be formed by (current_set \ {y}) U {k}
            # This is equivalent to checking if 1/y - 1/k can be formed by current_set \ {y, k}
            # IF 1/y - 1/k > 0.
            # If 1/y - 1/k <= 0, it's impossible (since sums are positive).
            # Note: 1/y - 1/k = (k-y)/(ky).
            # So if k > y, 1/y - 1/k > 0.
            # If k < y, 1/y - 1/k < 0, so no new subsum involving k can form 1/y 
            # because 1/k > 1/y, so adding 1/k to any positive sum makes it > 1/y.
            # Wait, we are checking if 1/y is a sum of a subset of S_new \ {y}.
            # The subset might include k.
            # If the subset includes k, then sum = 1/k + sum(1/z for z in T).
            # So 1/y = 1/k + sum(T) => sum(T) = 1/y - 1/k.
            # If 1/y - 1/k < 0, impossible.
            # If 1/y - 1/k == 0, then T must be empty? No, T is subset of S \ {y, k}.
            # If T is empty, sum is 0. So 1/y = 1/k. But y != k, so 1/y != 1/k.
            # So if k < y, we don't need to check.
            
            if k < y:
                continue
                
            diff = _R[y] - _R[k]
            if diff <= 0:
                continue
                
            elems_diff = [_R[x] for x in current_set if x != y and x != k]
            if _subset_sum_exists(diff, elems_diff):
                valid = False
                break
                
        if not valid:
            return backtrack(idx + 1, current_set, current_size) # Exclude k
            
        # If valid, recurse with k included
        return backtrack(idx + 1, new_set, current_size + 1)
        
    # Run backtracking to find a set larger than greedy
    # We start with an empty set and try to build up to size > max_size
    # This is a bit tricky because we don't know the target size.
    # We can just run the backtracking and let it find the max.
    # But the pruning `current_size + (24 - idx) <= max_size` requires `max_size` to be updated.
    # This works if we update `max_size` when we find a valid set.
    
    # However, the backtracking as written might be slow if it explores many invalid branches.
    # Let's try it.
    
    backtrack(0, [], 0)
    
    if not best_set:
        # Fallback
        return list(range(13, 25))
        
    return sorted(best_set)

if __name__ == "__main__":
    s = construct()
    print(s)
    print(len(s))
