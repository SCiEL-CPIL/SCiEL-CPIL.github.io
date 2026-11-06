import time
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = 85.0

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _is_subset_sum(target, elements):
    """
    Check if target can be formed by a sum of reciprocals of a non-empty subset of elements.
    elements is a list of ints.
    """
    if not elements:
        return False
    
    # Pruning: If sum of all reciprocals < target, return False
    # We sort elements to allow suffix sum pruning in DFS
    # Smallest x -> Largest 1/x
    elems = sorted(elements, key=lambda x: x)
    n = len(elems)
    
    # Precompute suffix sums of reciprocals
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + Fraction(1, elems[i])
        
    if suffix_sums[0] < target:
        return False
        
    # If any single element reciprocal equals target
    for x in elems:
        if Fraction(1, x) == target:
            return True

    # Quick check: if target < smallest reciprocal, return False
    if target < Fraction(1, elems[-1]):
        return False

    def dfs(idx, current_sum):
        if current_sum == target:
            return True
        if idx >= n:
            return False
        
        # Pruning: If current_sum + remaining_sum < target
        if current_sum + suffix_sums[idx] < target:
            return False
            
        # Option 1: Include elems[idx]
        # Only if adding it doesn't exceed target
        if current_sum + Fraction(1, elems[idx]) <= target:
            if dfs(idx + 1, current_sum + Fraction(1, elems[idx])):
                return True
        
        # Option 2: Exclude elems[idx]
        if dfs(idx + 1, current_sum):
            return True
            
        return False
        
    return dfs(0, Fraction(0))

def _can_add(k, selected):
    """
    Check if we can add k to the selected set.
    selected is a list of ints.
    """
    target_k = Fraction(1, k)
    
    # Check 1: Is 1/k a subset sum of reciprocals of selected?
    if _is_subset_sum(target_k, selected):
        return False
        
    # Check 2: For each x in selected, is (1/x - 1/k) a subset sum of reciprocals of selected \ {x}?
    for x in selected:
        # If 1/x <= 1/k, then 1/x - 1/k <= 0, so it cannot be a sum of positive reciprocals
        if x >= k: 
            # 1/x <= 1/k
            continue
            
        target_x = Fraction(1, x) - Fraction(1, k)
        remaining = [y for y in selected if y != x]
        if _is_subset_sum(target_x, remaining):
            return False
            
    return True

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_set = []
    best_size = 0
    
    # Elements to consider, ordered descending
    # We assume descending is a good heuristic for large sets
    elements = list(range(24, 0, -1))
    
    def dfs(idx, current_set):
        nonlocal best_set, best_size
        
        # Check deadline
        if _time_remaining() < 1.0:
            return
            
        # Pruning: If we can't beat the best size, return
        if len(current_set) + (24 - idx) <= best_size:
            return
            
        if idx == 24:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            return
            
        # Try including elements[idx]
        k = elements[idx]
        if _can_add(k, current_set):
            current_set.append(k)
            dfs(idx + 1, current_set)
            current_set.pop()
            
        # Try excluding elements[idx]
        dfs(idx + 1, current_set)

    # Start the search
    # We can also try a few different orderings if time permits, 
    # but for now, descending is strong.
    # Let's add a simple greedy initialization to give B&B a head start
    greedy_set = []
    for k in elements:
        if _can_add(k, greedy_set):
            greedy_set.append(k)
            
    if len(greedy_set) > best_size:
        best_size = len(greedy_set)
        best_set = greedy_set[:]
        
    # Run the DFS to improve
    dfs(0, [])
    
    return best_set

# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Result size: {len(result)}")
    print(f"Result: {sorted(result)}")
