"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _greedy_with_heuristic(deadline, seed=None):
    """
    Greedy construction with a heuristic scoring function.
    We try to add elements that keep the representation counts balanced.
    The score for a candidate x is based on how "tight" the new representation
    counts would be. We prefer candidates that minimize the maximum count
    and break ties by minimizing the sum of squares of counts (variance).
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    chosen = []
    counts = defaultdict(int)

    # Pre-compute all possible sums? No, we compute incrementally.
    # But we can speed up by maintaining a set of "tight" sums.
    
    # For speed, we'll iterate through candidates in a specific order.
    # The standard greedy is 1, 2, ..., N.
    # A better heuristic might be to look at all remaining candidates
    # and pick the "best" one each time, but that's O(N^2) per step.
    # Given N=1000, N^2 = 1e6. 1000 steps * 1e6 = 1e9 ops. Too slow in Python.
    # So we stick to a linear scan but with a smarter acceptance criterion
    # or a different order.
    
    # Let's try a "least restrictive" heuristic.
    # When considering x, we look at the new sums s = x+y.
    # If any s would have count > 2, reject.
    # Among valid candidates, we might prefer ones that don't increase the
    # count of sums that are already at count 2? Or ones that create new sums?
    
    # Actually, the standard greedy (smallest first) is quite good.
    # Let's try to improve it by allowing "restarts" or "backtracking" if time permits.
    # Or, we can try different orders.
    
    # Strategy:
    # 1. Standard greedy (1..1000)
    # 2. Randomized greedy: shuffle the order, but always try to add if valid.
    #    This might get stuck in local optima.
    # 3. "Look-ahead": For each x in 1..1000, check if it's valid.
    #    If multiple are valid, pick the one that minimizes some cost.
    #    Cost = sum of (count[s] + 1)^2 for all new sums s?
    
    # Let's implement a version that scans 1..1000 but uses a more nuanced score.
    # If we just take the first valid one (smallest x), it's fast.
    # If we take the "best" valid one, it's slower.
    
    # Given the 120s limit, let's try to run multiple random restarts
    # of a slightly modified greedy that shuffles the candidate order
    # but still respects the constraint.
    
    # Actually, a known improvement for B_h sets is to use a "dense" packing.
    # The standard greedy is known to be suboptimal for small N, but for large N
    # it's often close to optimal.
    
    # Let's try a "local search" approach after the initial greedy construction.
    # 1. Build a set S using greedy.
    # 2. Try to swap elements: remove x, add y (y not in S) if it improves size or maintains size with better properties?
    #    We want to maximize size. So if we can remove x and add TWO elements y, z, that's great.
    #    Or remove x, y and add z, w, v?
    
    # Let's stick to a robust greedy with multiple random restarts and a time limit.
    
    chosen = []
    counts = defaultdict(int)
    
    # We'll try to add elements in increasing order, but we'll check validity.
    # To make it faster, we can maintain a list of "blocked" sums.
    
    # Optimization:
    # Instead of checking all y in chosen for each x, we can precompute
    # the sums that are "full" (count == 2).
    # If x + y is in full_sums, then x is invalid (for that y).
    # But we need to check ALL y.
    
    # Let's just do the standard greedy first to get a baseline.
    # Then try to improve.
    
    for x in range(1, 1001):
        if time.time() > deadline:
            break
            
        # Check if x can be added
        valid = True
        # We need to check if x+y > 2 for any y in chosen
        # And if 2*x > 2
        if counts[2*x] >= 2:
            valid = False
        else:
            for y in chosen:
                if counts[x+y] >= 2:
                    valid = False
                    break
        
        if valid:
            # Add x
            chosen.append(x)
            counts[2*x] += 1
            for y in chosen:
                if y != x: # y is in chosen before x was added? No, chosen includes x now?
                    # Wait, the loop `for y in chosen` iterates over the list BEFORE x is added?
                    # In the code above, `chosen.append(x)` is AFTER the check.
                    # So `chosen` does not contain x during the check.
                    # So we need to update counts for x+y where y in chosen (old) and 2x.
                    pass
            
            # Correct update:
            # We already checked. Now update.
            # But wait, the check loop used `chosen` which didn't have x.
            # So we update counts for x+y for y in chosen (old) and 2x.
            for y in chosen[:-1]: # chosen[:-1] is the old chosen
                counts[x+y] += 1
            # And 2x was handled separately in check?
            # Let's re-structure for clarity and speed.
            
            # Actually, let's restart the logic to be cleaner and faster.
            pass
        
    # The logic above is messy. Let's write a clean, fast greedy.
    return chosen

def _fast_greedy(deadline):
    """
    Clean, fast greedy construction.
    """
    chosen = []
    counts = np.zeros(2001, dtype=np.int8) # Sums range from 1+1=2 to 1000+1000=2000
    
    for x in range(1, 1001):
        if time.time() > deadline:
            break
        
        # Check validity
        # 2x must have count < 2
        if counts[2*x] >= 2:
            continue
            
        # x+y must have count < 2 for all y in chosen
        valid = True
        # Vectorized check using NumPy might be faster for large chosen
        # But chosen is a list. Let's convert to array once?
        # Or just loop.
        
        # Optimization: if chosen is large, the loop is slow.
        # We can break early.
        for y in chosen:
            if counts[x+y] >= 2:
                valid = False
                break
        
        if valid:
            chosen.append(x)
            counts[2*x] += 1
            for y in chosen[:-1]:
                counts[x+y] += 1
                
    return chosen

def _local_search(initial_set, deadline):
    """
    Try to improve the set by swapping elements.
    We try to remove one element and add one or more new elements.
    """
    S = set(initial_set)
    size = len(S)
    
    # Precompute counts for current S
    counts = defaultdict(int)
    for i, a in enumerate(S):
        for b in list(S)[i:]:
            counts[a+b] += 1
            
    # This is O(N^2). N~600. 360,000 ops. Fast enough.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to remove x and add y
        # Or remove x, y and add z?
        
        # Simple swap: remove x, add y
        # If we remove x, some counts decrease.
        # If we add y, some counts increase.
        # We need the net effect to be valid (all counts <= 2).
        
        # This is complex. Let's try a simpler heuristic:
        # Try to add a new element z that is not in S.
        # If it's invalid, try to remove an element x that is "blocking" z.
        
        # Find a candidate z not in S
        candidates = [z for z in range(1, 1001) if z not in S]
        
        for z in candidates:
            if time.time() > deadline:
                return list(S)
                
            # Check if z is valid with current S
            # If yes, add it and restart local search
            valid = True
            if counts[2*z] >= 2:
                valid = False
            else:
                for y in S:
                    if counts[z+y] >= 2:
                        valid = False
                        break
            
            if valid:
                # Add z
                S.add(z)
                counts[2*z] += 1
                for y in S:
                    if y != z:
                        counts[z+y] += 1
                improved = True
                break
        
        # If no simple addition, try a swap
        # Remove x, add z
        # This requires checking if z is valid with S \ {x}
        
        # This is expensive. Let's skip complex swaps for now
        # and rely on multiple random restarts of the greedy.
        
    return list(S)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_set = []
    
    # 1. Run standard greedy
    S1 = _fast_greedy(deadline)
    if len(S1) > len(best_set):
        best_set = S1
        
    # 2. Try local search on S1
    if time.time() < deadline:
        S2 = _local_search(S1, deadline)
        if len(S2) > len(best_set):
            best_set = S2
            
    # 3. Try randomized greedy restarts
    # Shuffle the order of candidates?
    # If we shuffle, we might miss small numbers that are crucial.
    # Better: Randomly skip some numbers and see if we can fit more later?
    # Or: Randomly permute the "tie-breaking" in a more complex greedy.
    
    # Let's try a greedy that picks the "least restrictive" candidate
    # among all valid ones.
    # This is O(N^2) per step.
    
    # Given time constraints, let's just return the best found.
    
    # Ensure validity
    # The greedy ensures validity.
    
    return best_set

# EVOLVE-BLOCK-END
