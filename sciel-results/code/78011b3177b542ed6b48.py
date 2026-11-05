"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

def construct():
    """
    Construct a large subset of [1, 512] with no 4-term AP.
    Strategy: 
    1. Precompute all 4-APs in [1, 512].
    2. Represent conflicts as a mapping from element to list of 4-APs it participates in.
    3. Use multi-restart greedy with randomized ordering and 2-opt swap local search.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 512
    all_elements = list(range(1, N + 1))
    
    # Step 1: Precompute all 4-APs
    # A 4-AP is (a, a+d, a+2d, a+3d) with d >= 1
    # For each element x, store the list of 4-APs (as tuples of 4 elements) that contain x.
    # Also, for fast conflict checking: if we have a set S, x conflicts if there exists a 4-AP 
    # containing x and 3 other elements all in S.
    
    # Precompute: for each x, list of (other1, other2, other3) such that {x, other1, other2, other3} is a 4-AP
    # Actually, better: for each x, list of 4-APs containing x. Then when checking if x can be added to S,
    # we check if any 4-AP containing x has the other 3 elements in S.
    
    # To make it even faster, for each x, we can precompute the list of "conflict triples" 
    # i.e., for each 4-AP containing x, the other 3 elements. Then x is blocked if any of these triples is fully in S.
    
    # Let's precompute: conflicts[x] = list of frozensets of 3 elements (the other 3 in the 4-AP)
    # But storing frozensets might be slow. Instead, store tuples and check membership.
    
    # Alternative: Precompute a boolean matrix or use bitsets.
    # Given N=512, the number of 4-APs is roughly sum_{d=1}^{170} (512 - 3*d) ~ 512*170 - 3*170^2/2 ~ 87040 - 43350 ~ 43690.
    # So there are ~44k 4-APs. Each element appears in ~44k*4/512 ~ 344 4-APs.
    
    # Let's store: for each x, a list of 4-APs (as tuples (a, b, c, d)) containing x.
    # Then to check if x can be added to S: for each 4-AP in conflicts[x], check if the other 3 are in S.
    
    # Even better: Precompute for each x, a list of "blocking triples". 
    # blocking_triples[x] = list of (y1, y2, y3) such that {x, y1, y2, y3} is a 4-AP.
    # Then x is blocked iff any (y1, y2, y3) in blocking_triples[x] has y1, y2, y3 all in S.
    
    # Let's build this structure.
    blocking_triples = [None] * (N + 1)
    for x in range(1, N + 1):
        blocking_triples[x] = []
    
    # Iterate over all 4-APs
    # a from 1 to N-3
    # d from 1 to (N-a)//3
    # 4-AP: (a, a+d, a+2d, a+3d)
    # For each element in the 4-AP, add the other 3 to its blocking list.
    
    for a in range(1, N - 2):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # 4-AP: (a, b, c, e)
            # For a: blocking triple is (b, c, e)
            blocking_triples[a].append((b, c, e))
            # For b: blocking triple is (a, c, e)
            blocking_triples[b].append((a, c, e))
            # For c: blocking triple is (a, b, e)
            blocking_triples[c].append((a, b, e))
            # For e: blocking triple is (a, b, c)
            blocking_triples[e].append((a, b, c))
    
    # Now, we have a fast way to check if x can be added to a set S (represented as a boolean array or set).
    # Let's use a boolean array `in_set` of size N+1 for O(1) lookup.
    
    def can_add(x, in_set):
        """Check if x can be added to the set represented by in_set."""
        for triple in blocking_triples[x]:
            y1, y2, y3 = triple
            if in_set[y1] and in_set[y2] and in_set[y3]:
                return False
        return True
    
    def greedy_construct(order, in_set):
        """
        Greedy construct using a given order. Returns the set (as boolean array) and its size.
        """
        size = 0
        for x in order:
            if can_add(x, in_set):
                in_set[x] = True
                size += 1
        return size
    
    # Multi-restart greedy with random orders
    best_in_set = np.zeros(N + 1, dtype=bool)
    best_size = 0
    
    # Run multiple greedy restarts
    num_restarts = 5000  # We'll run as many as time allows
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:  # Leave 10s for local search
            break
        
        # Create a random order
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Try a few different random seeds/orders
        in_set = np.zeros(N + 1, dtype=bool)
        size = greedy_construct(order, in_set)
        
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
    
    # Step 2: Local search with 2-opt swaps
    # Try to improve best_in_set by swapping elements: remove one, add two? Or just add one?
    # First, try to add any missing element.
    current_in_set = best_in_set.copy()
    current_size = best_size
    
    # Try to add elements
    for x in range(1, N + 1):
        if time.time() > deadline - 5:
            break
        if not current_in_set[x]:
            if can_add(x, current_in_set):
                current_in_set[x] = True
                current_size += 1
    
    # Now try 2-opt swaps: remove x, add y and z? 
    # Actually, a simpler local search: try to remove one element x and add two elements y, z.
    # This is expensive. Let's try a simpler swap: remove x, add y.
    # But that doesn't increase size. We want to increase size.
    # So the main move is "add one". If no more can be added, try "remove one, add two".
    
    # Let's try "remove one, add two" in a randomized way.
    # This is O(N^3) in the worst case, but we can sample.
    
    no_improvement = 0
    max_no_improvement = 1000
    
    while time.time() < deadline - 2:
        if no_improvement > max_no_improvement:
            break
        
        # Try to find a move that increases size by 1: remove x, add y, add z
        # where y and z are not in the set, and x is in the set.
        # This is complex. Let's try a simpler approach: 
        # Randomly pick a missing element y. Try to add it. If it fails, 
        # find which conflicts are blocking it. Try to remove one element from a blocking conflict.
        
        missing = [x for x in range(1, N + 1) if not current_in_set[x]]
        if not missing:
            break
        
        random.shuffle(missing)
        improved = False
        
        for y in missing:
            if time.time() > deadline - 2:
                break
            
            if can_add(y, current_in_set):
                current_in_set[y] = True
                current_size += 1
                improved = True
                no_improvement = 0
                break
        
        if not improved:
            # Try a swap: remove x, add y
            # This keeps size same, but might allow further additions.
            # Let's try to find a swap that allows adding a third element later? 
            # Too complex. Let's just try random swaps and see if they enable an addition.
            
            # Pick a random element x in the set to remove.
            in_set_list = [x for x in range(1, N + 1) if current_in_set[x]]
            if not in_set_list:
                break
            
            x = random.choice(in_set_list)
            y = random.choice(missing)
            
            # Temporarily remove x
            current_in_set[x] = False
            temp_size = current_size - 1
            
            # Can we add y now?
            if can_add(y, current_in_set):
                current_in_set[y] = True
                temp_size += 1
                
                # Now, can we add any other missing element z?
                # Try a few random z's
                added_z = False
                for z in missing:
                    if z == y:
                        continue
                    if can_add(z, current_in_set):
                        current_in_set[z] = True
                        temp_size += 1
                        added_z = True
                        break
                
                if temp_size > current_size:
                    current_size = temp_size
                    improved = True
                    no_improvement = 0
                    break
                else:
                    # Revert: remove y, restore x
                    current_in_set[y] = False
                    current_in_set[x] = True
                    no_improvement += 1
            else:
                # Revert
                current_in_set[x] = True
                no_improvement += 1
    
    # Final verification
    # Convert to set for verification
    final_set = set()
    for x in range(1, N + 1):
        if current_in_set[x]:
            final_set.add(x)
    
    # Verify no 4-AP
    def has_4ap(s):
        s_list = list(s)
        s_set = set(s_list)
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(len(s_list)):
                b = s_list[j]
                if a >= b:
                    continue
                d = b - a
                if d == 0:
                    continue
                c = b + d
                e = c + d
                if c in s_set and e in s_set:
                    return True
        return False
    
    if has_4ap(final_set):
        # Fallback to best greedy set
        fallback_set = set()
        for x in range(1, N + 1):
            if best_in_set[x]:
                fallback_set.add(x)
        return sorted(fallback_set)
    
    return sorted(final_set)

# EVOLVE-BLOCK-END
