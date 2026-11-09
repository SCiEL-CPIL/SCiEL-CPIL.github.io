import time
import random

L = 5354228880  # lcm(1..24)

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    weights = [0] * 25
    for i in range(1, 25):
        weights[i] = L // i
        
    best_set = set()
    best_size = 0
    
    # Run randomized greedy with swaps
    # We will try to build a set by picking elements in random order.
    # If an element conflicts, we try to swap it with an existing element.
    
    while time.time() < deadline:
        elements = list(range(1, 25))
        random.shuffle(elements)
        
        S = set()
        sums = {0}
        
        for x in elements:
            wx = weights[x]
            
            if wx in sums:
                # Conflict. Try to find a z in S to remove such that wx is not in sums(S\{z})
                # To speed this up, we can just try a random z from S.
                # If removing z fixes it, we swap.
                # If S is small, we can try all z.
                # Let's try all z if len(S) < 10, else random sample.
                candidates = list(S)
                random.shuffle(candidates)
                
                swapped = False
                for z in candidates:
                    wz = weights[z]
                    # Check if wx is in sums of S\{z}
                    # We can compute this by removing wz from the contribution?
                    # It's easier to just recompute sums for S\{z} if S is small.
                    # But recomputing is O(2^|S|).
                    # Alternative: 
                    # sums_without_z = {s for s in sums if ...} ? No, subset sums are not easily reversible.
                    
                    # Since we are in a loop, let's just do the check.
                    # Optimization: If len(S) is large, this is slow.
                    # But we only do this if conflict occurs.
                    
                    # Let's try a cheaper check:
                    # If wx == wz, removing z doesn't help if there's another z' with same weight? 
                    # Weights are distinct for 1..24? 
                    # L/i are distinct.
                    
                    # Let's just recompute sums for S\{z} if len(S) <= 12.
                    if len(S) <= 12:
                        S_temp = S - {z}
                        sums_temp = {0}
                        for y in S_temp:
                            wy = weights[y]
                            new_sums_temp = set()
                            for s in sums_temp:
                                ns = s + wy
                                if ns <= L:
                                    new_sums_temp.add(ns)
                            sums_temp.update(new_sums_temp)
                        
                        if wx not in sums_temp:
                            # Swap z with x
                            S.remove(z)
                            S.add(x)
                            
                            # Update sums
                            # We need to update sums to reflect S.
                            # We already have sums_temp for S\{z}.
                            # Now add x to S_temp to get sums for new S.
                            new_sums = set()
                            for s in sums_temp:
                                ns = s + wx
                                if ns <= L:
                                    new_sums.add(ns)
                            sums_temp.update(new_sums)
                            sums = sums_temp
                            
                            swapped = True
                            break
                    
                if not swapped:
                    # Cannot swap, skip x
                    pass
            else:
                # No conflict, add x
                S.add(x)
                new_sums = set()
                for s in sums:
                    ns = s + wx
                    if ns <= L:
                        new_sums.add(ns)
                sums.update(new_sums)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
            # If we reach a high size, we can be more aggressive with time usage or break.
            # Max possible is 24, but realistically 10-12.
            if best_size >= 12:
                # We found a good solution. Let's keep searching a bit more to see if we can get 13.
                # But 13 is unlikely.
                pass

    return list(best_set)
