import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    n = 256
    # Weights are 1/i
    # Conflict: a, b, c form 3-AP if a+c=2b.
    
    # Helper to check if adding x to set S creates a 3-AP
    # S is a set of ints
    def can_add(x, S):
        # Check if x forms a 3-AP with any pair in S
        # If x is the largest: x = 2b - a => a = 2b - x. Need a in S, b in S, a < b < x.
        # If x is the middle: x = (a+c)/2 => c = 2x - a. Need a in S, c in S, a < x < c.
        # If x is the smallest: x = 2b - c => c = 2b - x. Need b in S, c in S, x < b < c.
        # Simpler: for any a, b in S, check if 2b-a == x.
        # This is O(|S|^2). |S| is at most ~100? 256^2 is 65k. Fast enough.
        # Even faster: iterate a in S. For each b in S.
        # Actually, we can just check:
        # 1. Is there a, b in S such that 2b - a == x?
        #    This implies a = 2b - x. So for each b in S, check if (2b - x) is in S.
        #    Also need a != b (non-constant AP). If a=b, then x=b, but x not in S.
        #    So if x not in S, and there exists b in S such that (2b - x) in S, then 3-AP exists.
        #    Wait, the triple is (a, b, x) or (x, b, a) or (b, x, a) etc.
        #    The condition "no 3-AP" means no distinct a,b,c in S with a+c=2b.
        #    If we add x, we need to ensure no a,b in S_old such that:
        #    1. a + x = 2b  => x = 2b - a. (x is largest or smallest? No, x is one end)
        #       If x is largest: a < b < x.
        #       If x is smallest: x < b < a.
        #    2. a + b = 2x  => x = (a+b)/2. (x is middle)
        #       Need a < x < b or b < x < a.
        
        # Efficient check:
        # For each b in S:
        #   a = 2*b - x
        #   if a in S and a != b: return False (x is an end)
        # For each a in S:
        #   b = (x + a) / 2
        #   if b is integer and b in S and b != a and b != x: return False (x is middle)
        
        # Note: If a=x, then 2b-x=x => b=x, but x not in S. So a!=x is guaranteed if x not in S.
        # So we just check if a in S.
        
        # Let's implement this O(|S|) check.
        for b in S:
            a = 2 * b - x
            if a in S:
                return False
        for a in S:
            # x is middle: a + c = 2x => c = 2x - a
            # We already covered the case where x is an end?
            # If x is an end, say x, b, a. Then x + a = 2b => a = 2b - x.
            # If x is middle, say a, x, c. Then a + c = 2x => c = 2x - a.
            # The first loop checks if x is an end (x, b, a or a, b, x).
            # The second loop checks if x is a middle (a, x, c).
            # So yes, both are needed.
            c = 2 * x - a
            if c in S:
                return False
        return True

    # Greedy initialization
    selected = set()
    for x in range(1, n + 1):
        if can_add(x, selected):
            selected.add(x)
            
    current_weight = sum(1.0 / x for x in selected)
    
    # Local Search
    # Strategy: Remove one element, then greedily fill the rest.
    # We try to remove each element and see if the greedy fill yields a better weight.
    
    # To speed up, we can prioritize removals of elements that are "blocking" many others.
    # But simple iteration is fine for N=256.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element
        for x in list(selected):
            if time.time() > deadline:
                break
            
            # Remove x
            selected.remove(x)
            
            # Greedy fill
            # We want to add numbers 1..256 not in selected, in order of increasing value (decreasing weight)
            # But we must ensure no 3-AP.
            # We can just iterate 1..256 and add if valid.
            # This is deterministic.
            
            # Optimization: We only need to add numbers that were not in the original set or x?
            # No, removing x might allow adding numbers that were previously blocked by x.
            # So we iterate all candidates 1..256.
            
            # To make it faster, we can just check candidates not in selected.
            # Since selected changes, we should be careful.
            
            # Let's create a copy to try the move?
            # Or modify in place and revert if worse.
            
            old_weight = current_weight
            # We already removed x, so weight is old_weight - 1/x
            # Let's just run the greedy fill on the current `selected`
            
            # We need to be careful: the greedy fill depends on the order.
            # Standard greedy: iterate 1 to N, add if valid.
            # Since we removed x, some numbers that were invalid before might be valid now.
            # We iterate 1 to N.
            
            # This is O(N * |S|).
            # We can optimize can_add to be faster if we maintain a sorted list or bitset, but set is fine.
            
            for y in range(1, n + 1):
                if y not in selected:
                    if can_add(y, selected):
                        selected.add(y)
                        
            new_weight = sum(1.0 / x for x in selected)
            
            if new_weight > old_weight + 1e-12:
                current_weight = new_weight
                improved = True
            else:
                # Revert
                # We need to restore the set to its state before removing x and filling.
                # This is tricky because we modified `selected` in place.
                # Better to work on a copy or store the old set.
                # Storing old set is O(|S|).
                
                # Let's refactor to use a copy for the trial.
                # This is safer.
                pass
                
        # The in-place modification is dangerous for reverting.
        # Let's restart the loop logic to use copies.
        
    # The above loop is flawed because of in-place modification without proper revert.
    # Let's rewrite the local search properly.
    
    # Reset to initial greedy
    selected = set()
    for x in range(1, n + 1):
        if can_add(x, selected):
            selected.add(x)
    current_weight = sum(1.0 / x for x in selected)
    
    # Local Search with Copies
    while time.time() < deadline:
        improved = False
        # Try removing each element
        for x in list(selected):
            if time.time() > deadline:
                return sorted(selected)
            
            # Trial set
            trial_set = set(selected)
            trial_set.remove(x)
            
            # Greedy fill on trial_set
            for y in range(1, n + 1):
                if y not in trial_set:
                    if can_add(y, trial_set):
                        trial_set.add(y)
                        
            trial_weight = sum(1.0 / x for x in trial_set)
            
            if trial_weight > current_weight + 1e-12:
                selected = trial_set
                current_weight = trial_weight
                improved = True
                
        if not improved:
            break
            
    return sorted(selected)
