import time
import itertools


def construct():
    start_time = time.time()
    deadline = start_time + 70  # 70s safety margin (90s limit)
    
    # The key insight: we want the largest subset of [1,100] with no subset summing to 200.
    # 
    # A simple construction: take all numbers > 100... but we only have [1,100].
    # 
    # Better: if we take all numbers from 51 to 100 (50 numbers), can any subset sum to 200?
    # The minimum sum of any non-empty subset is 51. The maximum is 5050-1275=3775... wait.
    # Sum of 51..100 = (51+100)*50/2 = 3775.
    # Can a subset of {51,...,100} sum to 200? 
    # Minimum element is 51, so we need at most floor(200/51) = 3 elements.
    # With 1 element: need 200, not in set.
    # With 2 elements: need a+b=200, a,b in [51,100]. E.g., 51+149 (no), 60+140 (no), 
    #   100+100 (can't use twice). Actually 51+149 no... max is 100+100=200 but can't repeat.
    #   99+101 no. So 2 elements: a+b=200, both in [51,100], distinct. 
    #   100+100 not allowed (distinct). 99+101 no. So no 2-element subset sums to 200.
    # With 3 elements: a+b+c=200, all in [51,100]. Min sum = 51+52+53=156, max=98+99+100=297.
    #   E.g., 51+52+97=200. Yes! So {51,...,100} is NOT valid.
    #
    # Let's think differently. Take all numbers > 200/2 = 100. But max is 100.
    #
    # Take numbers in (100, 200]? None exist in [1,100].
    #
    # What about taking all numbers from 67 to 100? That's 34 numbers.
    # Min 2-element sum: 67+68=135. Min 3-element: 67+68+69=204 > 200.
    # So only 1 or 2 elements can potentially sum to 200.
    # 1 element: 200 not in [67,100].
    # 2 elements: a+b=200, a,b in [67,100]. 100+100 no. 99+101 no. Max distinct sum = 99+100=199 < 200.
    # So {67,...,100} has NO subset summing to 200! Cardinality = 34.
    #
    # Can we do better? What about {66,...,100}? 35 numbers.
    # 2 elements: 66+134 no. Max 2-sum = 99+100=199. Still fine for 2 elements.
    # But 3 elements: 66+67+68=201 > 200. Still fine.
    # Wait, 66+67+67? No, distinct. 66+67+68=201. So min 3-sum is 201 > 200.
    # So {66,...,100} also works! Cardinality = 35.
    #
    # {65,...,100}? 36 numbers.
    # 3 elements: 65+66+67=198 <= 200. Can we get exactly 200?
    # 65+66+69=200. Yes! So {65,...,100} is invalid.
    #
    # So the largest interval is {66,...,100} with 35 elements.
    #
    # Can we get more than 35 by a non-interval subset?
    # We could potentially include some numbers below 66 if we exclude some that conflict.
    #
    # Let's try a greedy approach with bitset to find the maximum.
    # 
    # Actually, let me think about this more carefully. The problem is equivalent to:
    # Find the maximum size subset S of [1,100] such that 200 is not in the subset sum set of S.
    #
    # A known result: for target T, the largest subset of [1,n] avoiding sum T...
    # 
    # Let me just use a smart greedy with bitsets.

    MASK = (1 << 201) - 1
    
    def can_add(reachable, x):
        """Check if adding x to a set with given reachable sums doesn't create sum 200."""
        new_reachable = reachable | (reachable << x)
        return not (new_reachable & (1 << 200)), new_reachable & MASK
    
    # Strategy: Try to build the largest possible set.
    # Start with the known valid set {66,...,100} (35 elements).
    # Then try to add elements from 1 to 65 in some order.
    
    # First, let's verify {66,...,100} is valid and compute its reachable set
    reachable = 1
    for x in range(66, 101):
        reachable = (reachable | (reachable << x)) & MASK
    
    assert not (reachable & (1 << 200)), "Base set {66..100} should be valid"
    
    base_selected = list(range(66, 101))
    base_size = len(base_selected)
    
    # Now try to add elements from 1 to 65
    # Try different orderings to maximize additions
    candidates = list(range(1, 66))
    
    best_selected = base_selected[:]
    best_reachable = reachable
    
    # Try multiple orderings
    orderings = [
        candidates,                          # ascending
        candidates[::-1],                    # descending
        sorted(candidates, key=lambda x: x % 7),  # by mod 7
        sorted(candidates, key=lambda x: -x % 13),# by neg mod 13
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        
        sel = base_selected[:]
        reach = reachable
        added = 0
        
        for x in order:
            if time.time() > deadline:
                break
            ok, new_reach = can_add(reach, x)
            if ok:
                sel.append(x)
                reach = new_reach
                added += 1
        
        if len(sel) > len(best_selected):
            best_selected = sel
            best_reachable = reach
    
    # Try local improvement: swap elements
    if time.time() < deadline:
        selected_set = set(best_selected)
        excluded = [x for x in range(1, 101) if x not in selected_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in excluded[:]:
                if time.time() > deadline:
                    break
                if x in selected_set:
                    continue
                
                # Try adding x directly
                ok, new_reach = can_add(best_reachable, x)
                if ok:
                    best_selected.append(x)
                    best_reachable = new_reach
                    selected_set.add(x)
                    excluded.remove(x)
                    improved = True
                    continue
                
                # Try swapping: remove y, add x
                for y in best_selected[:]:
                    if time.time() > deadline:
                        break
                    # Remove y, add x
                    new_sel = [z for z in best_selected if z != y]
                    new_reach = 1
                    for z in new_sel:
                        new_reach = (new_reach | (new_reach << z)) & MASK
                    ok, new_reach_with_x = can_add(new_reach, x)
                    if ok:
                        best_selected = new_sel + [x]
                        best_reachable = new_reach_with_x
                        selected_set.discard(y)
                        selected_set.add(x)
                        if y not in excluded:
                            excluded.append(y)
                        if x in excluded:
                            excluded.remove(x)
                        improved = True
                        break
    
    # Final verification
    final_reachable = 1
    for x in best_selected:
        final_reachable = (final_reachable | (final_reachable << x)) & MASK
    
    # Ensure validity - if somehow invalid, fall back to base set
    if final_reachable & (1 << 200):
        best_selected = list(range(66, 101))
    
    return best_selected
