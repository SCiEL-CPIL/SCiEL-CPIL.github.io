import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s kill

    # 1. Precompute forbidden differences (non-zero squares mod 205)
    forbidden = set()
    for x in range(1, m):
        sq = (x * x) % m
        if sq != 0:
            forbidden.add(sq)

    # 2. Build adjacency bitmasks: adj[u] has bit v set iff u and v conflict
    adj = [0] * m
    for u in range(m):
        mask = 0
        for v in range(m):
            if u != v:
                d = (u - v) % m
                if d in forbidden:
                    mask |= (1 << v)
        adj[u] = mask

    # Helper to check if a vertex can be added to a set represented by a bitmask
    # It checks if vertex v conflicts with any vertex in the set
    def can_add(v, set_mask):
        # Check if v is already in set
        if (set_mask >> v) & 1:
            return False
        # Check if any neighbor of v is in set
        # If adj[v] & set_mask is not 0, there is a conflict
        return (adj[v] & set_mask) == 0

    def greedy_construct(forbidden_mask=0):
        """
        Construct a maximal independent set.
        forbidden_mask: vertices that cannot be added.
        """
        current_mask = 0
        available = ((1 << m) - 1) & ~forbidden_mask
        while available:
            # Pick a random vertex from available to diversify
            # Or pick min degree? Random is good for diversity in restarts.
            # Let's pick a random vertex from available
            # To pick random bit:
            # 1. Get list of set bits? Slow.
            # 2. Use random integer and mask?
            
            # Efficient random bit selection from mask
            # Convert to list of indices for simplicity since m=205 is small
            # Actually, let's just iterate and pick the first one? No, that's deterministic.
            # Let's pick a random index from the set bits.
            
            # Optimization: Precompute list of indices? No, mask changes.
            # Just use a simple loop to find a random candidate.
            # Since m is small (205), we can just pick a random v in 0..m-1 
            # and check if it's in available.
            
            found = False
            while not found:
                v = random.randint(0, m - 1)
                if (available >> v) & 1:
                    found = True
                    
            current_mask |= (1 << v)
            # Remove v and its neighbors from available
            available &= ~(1 << v)
            available &= ~adj[v]
            
        return current_mask

    def local_search(initial_mask, max_iters=500):
        """
        Perform local search to improve the independent set.
        Uses 1-swap and 2-swap moves.
        """
        current_mask = initial_mask
        current_size = current_mask.bit_count()
        no_improvement_count = 0
        iterations = 0
        
        while iterations < max_iters and time.time() < deadline:
            iterations += 1
            improved = False
            
            # Try 1-swap: Remove one vertex, add one or more
            # Or simply: Try to add a random vertex from complement?
            # If we add a vertex, it must not conflict with any in set.
            # If we remove a vertex, we might be able to add more.
            
            # Strategy: 
            # 1. Try to add a random vertex from the complement that is valid.
            #    If found, add it.
            # 2. If no simple add, try a swap: Remove a random vertex v from set,
            #    then try to add a random vertex u from complement that is now valid
            #    (u might have conflicted with v, but not others).
            
            # Get list of vertices in set and not in set
            set_verts = [i for i in range(m) if (current_mask >> i) & 1]
            comp_verts = [i for i in range(m) if not (current_mask >> i) & 1]
            
            if not set_verts or not comp_verts:
                break

            # Try to add a valid vertex from complement
            # Shuffle complement to pick random candidates
            random.shuffle(comp_verts)
            added = False
            for u in comp_verts:
                if can_add(u, current_mask):
                    current_mask |= (1 << u)
                    current_size += 1
                    improved = True
                    added = True
                    break
            
            if added:
                no_improvement_count = 0
                continue
            
            # Try 1-swap: Remove v, add u
            # We need u to be valid in (current_mask \ {v})
            # And u != v (u is in comp, v in set, so u!=v always)
            # And u should ideally "replace" v, i.e., u conflicts with v? 
            # Not necessarily, but if u didn't conflict with v, it would have been added in step 1.
            # So u MUST conflict with v to be a candidate for swap (otherwise it was addable).
            
            random.shuffle(set_verts)
            swapped = False
            for v in set_verts:
                # Try to find a u in comp that is valid if v is removed
                # u is valid in (current_mask & ~(1<<v))
                # This is expensive to check for all u.
                # Optimization: Only consider u that conflict with v.
                # If u does not conflict with v, it was already checked in step 1?
                # No, step 1 checked if u is valid in current_mask.
                # If u is valid in current_mask, it would have been added.
                # So if we are here, no u in comp is valid in current_mask.
                # Therefore, any u we consider for swap MUST have a conflict with some vertex in current_mask.
                # Specifically, if we remove v, u becomes valid if its ONLY conflict in current_mask was v.
                
                # So we look for u in comp such that (adj[u] & current_mask) == (1 << v)
                
                # This is a strong condition. Let's check candidates.
                # To speed up, we can just iterate through a few random u's from comp.
                # Or, better: iterate through neighbors of v in comp?
                # If u is not a neighbor of v, then adj[u] & current_mask is a subset of adj[u] & (current_mask \ {v}).
                # If adj[u] & current_mask is not 0, and v is not in adj[u], then adj[u] & (current_mask \ {v}) is also not 0.
                # So u must be a neighbor of v to have a chance of becoming valid after removing v.
                
                # So candidates for u are: comp_verts that are neighbors of v.
                # i.e., u in comp_verts such that (adj[v] >> u) & 1
                # AND (adj[u] & current_mask) == (1 << v)
                
                candidates = []
                for u in comp_verts:
                    if (adj[v] >> u) & 1:
                        # Check if v is the only conflict
                        conflicts = adj[u] & current_mask
                        if conflicts == (1 << v):
                            candidates.append(u)
                
                if candidates:
                    u = random.choice(candidates)
                    # Perform swap
                    current_mask &= ~(1 << v)
                    current_mask |= (1 << u)
                    # Size remains same
                    improved = True
                    swapped = True
                    break
            
            if swapped:
                no_improvement_count = 0
                continue
            
            # If no improvement
            no_improvement_count += 1
            if no_improvement_count > 50:
                break
                
        return current_mask, current_size

    best_mask = 0
    best_size = 0

    # Run multiple rounds of Greedy + Local Search
    while time.time() < deadline - 10:
        # Randomized Greedy
        init_mask = greedy_construct()
        init_size = init_mask.bit_count()
        
        # Local Search
        new_mask, new_size = local_search(init_mask)
        
        if new_size > best_size:
            best_size = new_size
            best_mask = new_mask

    # Final check: If we still have time, try to polish the best solution
    if time.time() < deadline - 5:
        # Run local search again on the best found
        polished_mask, polished_size = local_search(best_mask, max_iters=1000)
        if polished_size > best_size:
            best_mask = polished_mask
            best_size = polished_size

    # Extract result
    result = []
    for i in range(m):
        if (best_mask >> i) & 1:
            result.append(i)

    if not result:
        # Fallback to a single valid element
        result = [0]

    return np.array(result, dtype=np.int64)
