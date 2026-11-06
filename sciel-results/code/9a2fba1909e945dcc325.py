import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def can_add(x, S_set):
        """Check if x can be added to S_set while maintaining sum-free property."""
        # x + a = b for a, b in S
        # a + x = b for a, b in S
        # x + x = b for b in S
        # a + b = x for a, b in S
        
        # Check 1: x + a in S or a + x in S
        for a in S_set:
            if a + x in S_set:
                return False
            # Note: a+x == x+a, so one loop is enough for this direction
            # But we also need to check if x is the sum of two elements in S
            # This is checked below.
            
        # Check 2: 2x in S
        if 2 * x in S_set:
            return False
            
        # Check 3: a + b = x for a, b in S
        # This is O(|S|^2). 
        # Optimization: If S is small, it's fine. If S is large, it's slow.
        # However, we can break early.
        S_list = list(S_set)
        for i in range(len(S_list)):
            a = S_list[i]
            for b in S_list[i:]: # a+b is commutative, and a,b distinct or same
                if a + b == x:
                    return False
        return True

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def solve_instance(S):
        S_set = set(S)
        odds = [x for x in S if x % 2 != 0]
        evens = [x for x in S if x % 2 == 0]
        
        current_set = set(odds)
        
        # Phase 1: Add safe evens
        # Sort evens by some heuristic? Maybe absolute value?
        # We want to add as many as possible.
        # If we add an even, it might block other evens.
        # Greedy: Add even if it doesn't conflict with current set.
        
        # To make this better, we can try multiple orders or just one.
        # Given the time limit, a single pass is safe.
        # But we can do a second pass after adding some, as removing conflicts isn't done, 
        # so adding more might be possible if the first additions didn't block them?
        # Actually, adding an element only adds constraints, never removes them.
        # So if an even couldn't be added, it still can't be added later.
        # So one pass is sufficient for "add if possible".
        
        for e in evens:
            if time.time() > deadline:
                break
            if can_add(e, current_set):
                current_set.add(e)
                
        # Phase 2: Local Search (Swaps)
        # Try to remove 1 element and add 2 elements.
        # This is complex. 
        # Simpler: Try to remove 1 element and add 1 element (no size gain, but might unlock?).
        # No, we want size gain.
        
        # Let's try a simple hill climbing:
        # For each element in current_set, try removing it.
        # Then try to add as many evens as possible from the remaining evens.
        # If size increases, keep it.
        
        best_set = current_set
        best_size = len(current_set)
        
        # Candidate elements to remove: mostly odds, as evens were added greedily.
        # Removing an even might allow another even? 
        # If e1 and e2 conflicted, removing e1 allows e2. Size stays same.
        # Removing an odd might allow multiple evens.
        
        # To save time, we only try removing odds.
        for o in list(odds):
            if time.time() > deadline:
                break
            if o not in best_set:
                continue
                
            test_set = set(best_set)
            test_set.remove(o)
            
            # Try to add evens that were previously rejected or not tried?
            # We should try all evens not in test_set.
            added_count = 0
            for e in evens:
                if time.time() > deadline:
                    break
                if e in test_set:
                    continue
                if can_add(e, test_set):
                    test_set.add(e)
                    added_count += 1
            
            if len(test_set) > best_size:
                best_set = test_set
                best_size = len(test_set)
                
        return list(best_set)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        
        # Fallback if time is tight
        if time.time() > deadline - 10:
             results.append([x for x in S if x % 2])
             continue
             
        res = solve_instance(S)
        
        # Verify
        if not is_sum_free(res):
            # Fallback to parity
            res = [x for x in S if x % 2]
            
        results.append(res)
        
    return results
