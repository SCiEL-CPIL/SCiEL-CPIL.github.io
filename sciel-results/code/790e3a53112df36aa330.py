"""
Combines the structural insight of focusing on the upper half of the interval 
{51, ..., 100} (where subset sums grow fast, limiting the depth of forbidden 
subsets to size 3) with an efficient hitting-set removal strategy to break 
forbidden triplets, followed by a greedy addition phase for the lower half 
{1, ..., 50} to maximize cardinality.
"""
import time

TARGET = 200
N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Phase 1: Initialize with upper half {51, ..., 100}
    # In this range, min 4-sum is 51+52+53+54 = 210 > 200.
    # So only subsets of size 1, 2, 3 can sum to 200.
    # Size 1: max 100 < 200.
    # Size 2: max 100+99 = 199 < 200.
    # Size 3: Possible.
    
    upper_half = list(range(51, 101))
    selected = set(upper_half)
    
    # Find all forbidden triplets in upper_half
    forbidden_triplets = []
    n = len(upper_half)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if upper_half[i] + upper_half[j] + upper_half[k] == TARGET:
                    forbidden_triplets.append((i, j, k))
    
    # Phase 2: Greedy Hitting Set to remove elements from upper_half
    # We want to remove a set of indices such that every triplet is hit.
    in_set = [True] * n
    active_triplets = forbidden_triplets
    
    while active_triplets:
        if time.time() > deadline:
            break
        # Count frequency of each index in active_triplets
        freq = [0] * n
        for t in active_triplets:
            for idx in t:
                if in_set[idx]:
                    freq[idx] += 1
        
        # Find index with max frequency
        max_idx = -1
        max_freq = -1
        for i in range(n):
            if in_set[i] and freq[i] > max_freq:
                max_freq = freq[i]
                max_idx = i
        
        if max_idx == -1:
            break
            
        # Remove this index
        in_set[max_idx] = False
        selected.discard(upper_half[max_idx])
        
        # Update active_triplets: remove any triplet containing max_idx
        active_triplets = [t for t in active_triplets if max_idx not in t]
        
    # Phase 3: Calculate reachable sums for the current selected set
    # Use a set of sums
    reachable = {0}
    for x in selected:
        new_sums = set()
        for s in reachable:
            ns = s + x
            if ns <= TARGET:
                new_sums.add(ns)
        reachable |= new_sums
        
    # Verify no 200
    if TARGET in reachable:
        # Fallback: should not happen if hitting set worked correctly
        # But just in case, return a known valid small set
        return [1]

    # Phase 4: Try to add elements from lower half {1, ..., 50}
    # Adding x is feasible if (TARGET - x) is NOT in reachable.
    # If we add x, we update reachable: reachable = reachable | {s+x for s in reachable if s+x <= TARGET}
    
    lower_half = list(range(1, 51))
    # Shuffle to try diverse orders? Or sorted? 
    # Smaller elements allow more combinations, so they are "riskier".
    # Larger elements in lower half (49, 50) are safer to add first?
    # Actually, adding ANY element is good.
    # Let's try adding in descending order of "safety" or just random.
    # Descending order of value: 50, 49, ...
    # Why? Because 200-50 = 150. Is 150 in reachable?
    # If we add 1, we check 199.
    
    # Let's iterate through all candidates 1..50
    # We can do multiple passes to ensure we add everything possible
    changed = True
    while changed:
        changed = False
        for x in range(50, 0, -1): # Try larger lower-half elements first
            if x in selected:
                continue
            # Check feasibility
            if (TARGET - x) not in reachable:
                # Add x
                selected.add(x)
                # Update reachable
                new_sums = set()
                for s in reachable:
                    ns = s + x
                    if ns <= TARGET:
                        new_sums.add(ns)
                reachable |= new_sums
                changed = True
                if time.time() > deadline:
                    break
        if time.time() > deadline:
            break

    # Final verification
    if TARGET in reachable:
        # This should not happen, but if it does, we have a bug in the logic above
        # Fallback to a simple valid set
        return [100, 99, 98, 97, 96] # Check: 96+97+98=291. 96+97+99=292. 
        # Actually, just return the upper half minus the hits, which was verified.
        # But if we added elements and broke it, we should revert.
        # The logic `if (TARGET - x) not in reachable` ensures that adding x 
        # does NOT create a sum of TARGET using x and any subset of the previous set.
        # Since the previous set was valid, the new set is valid.
        # So this check is redundant but safe.

    return list(selected)
