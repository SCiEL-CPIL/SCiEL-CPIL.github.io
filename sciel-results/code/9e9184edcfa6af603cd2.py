# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _try_add(x, chosen, counts):
    """Check if x can be added to chosen without violating B2[2] condition."""
    # Check 2*x
    if counts.get(2 * x, 0) >= 2:
        return False
    
    # Check x + y for all y in chosen where y < x
    # Note: chosen is a list, we can iterate.
    # Optimization: if chosen is sorted, we can break early? 
    # But x+y increases with y. 
    # The critical constraint is that counts[s] must be < 2 for all new s.
    for y in chosen:
        if y >= x:
            break
        s = x + y
        if counts.get(s, 0) >= 2:
            return False
    return True

def _greedy_construct(order):
    """
    Construct a B2[2] set by processing candidates in the given order.
    Returns the set as a list.
    """
    chosen = []
    counts = {}
    
    for x in order:
        if _try_add(x, chosen, counts):
            chosen.append(x)
            # Update counts
            counts[2 * x] = counts.get(2 * x, 0) + 1
            for y in chosen[:-1]: # chosen[:-1] are the previous elements, all < x? 
                # Wait, chosen is built in the order of 'order', not necessarily sorted.
                # The condition a <= b implies we consider pairs {a,b}.
                # My _try_add logic assumed chosen contains elements < x?
                # No, _try_add iterates `for y in chosen: if y >= x: break`.
                # This assumes `chosen` is sorted.
                # If `order` is random, `chosen` is NOT sorted.
                # I must sort `chosen` or iterate all `y`.
                # Iterating all `y` is O(N). N=1000. 1000*1000 = 1M ops.
                # Let's just iterate all y in chosen to be safe and correct.
                pass 
            
            # Correct update logic:
            # We need to update counts for all pairs (x, y) with y in chosen AND y != x.
            # Since x is new, the new pairs are (x, y) for all y in chosen.
            # The sum is x + y.
            # We must ensure we don't double count or miss.
            # The representation count for sum s is the number of pairs {a,b} in S such that a+b=s.
            # When we add x, we add pairs {x, y} for all y in S_old.
            # So for each y in S_old, we increment counts[x+y].
            
            for y in chosen[:-1]: # Wait, chosen[:-1] is the list BEFORE appending x? 
                # I appended x to chosen already in the code above? 
                # No, in the draft above I did `chosen.append(x)` then `for y in chosen[:-1]`.
                # This is correct if I want to pair x with all PREVIOUS elements.
                # But what if y > x? The pair is {x, y}. Sum is x+y.
                # The check in _try_add must also check all y, not just y < x.
                # Let's fix _try_add to check all y.
                pass
            
            # Let's restart the logic in the code block below.
            pass
            
    return chosen

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    best_set = []
    
    # Initial deterministic run (seed behavior)
    initial_order = list(range(1, 1001))
    # We need a robust greedy that works with unsorted input or maintains sortedness.
    # Simpler: Always maintain `chosen` sorted.
    # If we insert x into sorted `chosen`, we can do binary search or just linear scan.
    
    def _greedy_sorted(order):
        chosen = []
        counts = {}
        for x in order:
            # Check validity
            valid = True
            # Check 2x
            if counts.get(2 * x, 0) >= 2:
                valid = False
            else:
                # Check x + y for all y in chosen
                for y in chosen:
                    s = x + y
                    if counts.get(s, 0) >= 2:
                        valid = False
                        break
                # Also need to check if x+y collides with other new sums?
                # No, x+y1 != x+y2 for y1!=y2.
                # And x+y != 2x unless y=x (not in chosen).
                # So checking against existing counts is sufficient.
            
            if valid:
                # Insert x into chosen (keep sorted for potential future optimizations, 
                # though for the check we iterate all y anyway)
                # Simple insertion
                # Since we just need to iterate all y, order in `chosen` doesn't matter for correctness of update.
                # But keeping it sorted might help if we add heuristics later.
                chosen.append(x)
                # Update counts
                counts[2 * x] = counts.get(2 * x, 0) + 1
                for y in chosen[:-1]:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                
        return chosen

    # Run initial
    best_set = _greedy_sorted(initial_order)
    best_size = len(best_set)
    
    # Randomized restarts
    random.seed(42)
    while time.time() < deadline:
        # Random permutation
        order = list(range(1, 1001))
        random.shuffle(order)
        
        # Optional: Bias towards larger numbers? Or smaller?
        # Smaller numbers create smaller sums, which are more likely to collide?
        # Actually, sums of small numbers are in the "dense" region 2..2000.
        # Sums of large numbers are in 2000..2000.
        # The constraint is local.
        
        current_set = _greedy_sorted(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return best_set
# EVOLVE-BLOCK-END
