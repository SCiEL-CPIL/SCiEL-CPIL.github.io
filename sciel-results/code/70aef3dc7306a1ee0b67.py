"""Program for large reciprocal-subsum-free sets using backtracking."""
import time
from fractions import Fraction
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _can_add(k, selected):
    """
    Check if we can add k to the selected set.
    Condition: For every x in selected, 1/x must NOT be equal to the sum of reciprocals
    of any non-empty subset of (selected \ {x}) U {k}.
    
    When we add k to S to form S' = S U {k}:
    1. We must check if 1/k is a sum of reciprocals of a subset of S. 
    2. For each existing x in S, we must check if 1/x is a sum of reciprocals of a 
       subset of S' \ {x}.
       S' \ {x} = (S \ {x}) U {k}.
       The subsets of S' \ {x} are either subsets of S \ {x} (already safe) OR subsets 
       that include k.
       So we only need to check if 1/x is a sum of reciprocals of a subset T of S \ {x} 
       such that 1/x = sum(1/y for y in T) + 1/k.
       This is equivalent to: sum(1/y for y in T) = 1/x - 1/k.
    """
    target_k = Fraction(1, k)
    
    # Check 1: Is 1/k a subset sum of reciprocals of S?
    if _is_subset_sum(target_k, selected):
        return False
        
    # Check 2: For each x in S, is (1/x - 1/k) a subset sum of reciprocals of S \ {x}?
    for x in selected:
        target_x = Fraction(1, x) - Fraction(1, k)
        if target_x <= 0:
            continue
        
        remaining = [y for y in selected if y != x]
        if _is_subset_sum(target_x, remaining):
            return False
            
    return True

def _is_subset_sum(target, elements):
    """
    Check if target can be formed by a sum of reciprocals of a non-empty subset of elements.
    Elements are integers.
    """
    if not elements:
        return False
    
    # Pruning: If sum of all reciprocals < target, return False
    total = sum(Fraction(1, x) for x in elements)
    if total < target:
        return False
        
    # Pruning: If any single element reciprocal equals target, return True
    for x in elements:
        if Fraction(1, x) == target:
            return True
            
    # If target < min(1/x), return False
    min_recip = min(Fraction(1, x) for x in elements)
    if target < min_recip:
        return False
        
    elems = sorted(elements, key=lambda x: x) # Smallest x first -> Largest 1/x first
    n = len(elems)
    
    # Precompute suffix sums for pruning
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + Fraction(1, elems[i])
    
    def dfs(idx, current_sum):
        if current_sum == target:
            return True
        if idx >= n:
            return False
        
        # Pruning: If current_sum + suffix_sums[idx] < target, return False
        if current_sum + suffix_sums[idx] < target:
            return False
        
        # Option 1: Include elems[idx]
        new_sum = current_sum + Fraction(1, elems[idx])
        if new_sum <= target:
            if dfs(idx + 1, new_sum):
                return True
        
        # Option 2: Exclude elems[idx]
        if dfs(idx + 1, current_sum):
            return True
            
        return False
        
    return dfs(0, Fraction(0))

def _validate_set(S):
    """
    Verify that S is a valid reciprocal-subsum-free set.
    """
    S = set(S)
    for x in S:
        others = S - {x}
        # Check all non-empty subsets of others
        # For efficiency, we can just use _is_subset_sum
        if _is_subset_sum(Fraction(1, x), list(others)):
            return False
    return True

def construct():
    """
    Construct the largest subset of [1, 24] that is reciprocal-subsum-free.
    Strategy: Greedy descending construction followed by local search (add/swap).
    """
    S = []
    candidates = list(range(24, 0, -1)) # 24 down to 1
    
    # Phase 1: Greedy descending
    for c in candidates:
        if _time_remaining() < 10.0:
            break
        if _can_add(c, S):
            S.append(c)
            
    best_set = set(S)
    best_size = len(S)
    
    # Phase 2: Local search to try to increase cardinality
    # We can try to add elements that were rejected, potentially by removing others?
    # Or just try to add more if the greedy order missed some due to ordering?
    # Actually, if we process 24..1, and reject some small number, it's because of conflict with larger ones.
    # Removing a larger one to add two smaller ones could increase size.
    
    # Let's try a simple hill-climbing:
    # 1. Try to add any element not in S.
    # 2. If that fails, try swapping one element in S with one not in S, and then try to add more.
    
    # Since the search space is small (24 elements), we can be a bit more exhaustive if time permits.
    
    # Let's try to add elements in increasing order of "conflict potential" or just random?
    # Let's stick to a simple improvement loop.
    
    improved = True
    while improved and _time_remaining() > 20.0:
        improved = False
        # Try to add any element
        for c in range(1, 25):
            if c not in best_set:
                if _can_add(c, list(best_set)):
                    best_set.add(c)
                    improved = True
                    best_size += 1
                    break # Restart loop with new set
        
        # If no simple addition, try a swap?
        # This is complex. Let's just do addition for now.
        # The greedy phase should have added most safe ones.
        # The local search might catch cases where order mattered?
        # Actually, if _can_add is correct, the order of addition in Phase 1 doesn't matter for the final validity, 
        # but it matters for which set we end up with (greedy trap).
        # If we can't add anything, we are at a local maximum for "add only".
        
    # To potentially escape local maxima, we could try random restarts of the greedy phase.
    # But given the time budget, let's see if we can do a few quick restarts with different orderings.
    
    # Try a few random orderings if time permits
    import random
    random.seed(42)
    
    for _ in range(5):
        if _time_remaining() < 10.0:
            break
        
        order = list(range(1, 25))
        random.shuffle(order)
        
        S_temp = []
        for c in order:
            if _can_add(c, S_temp):
                S_temp.append(c)
                
        if len(S_temp) > best_size:
            best_set = set(S_temp)
            best_size = len(S_temp)
            
    return list(best_set)

# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Result size: {len(result)}")
    print(f"Result: {sorted(result)}")
    print(f"Valid: {_validate_set(result)}")
