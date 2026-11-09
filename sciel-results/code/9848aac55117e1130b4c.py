import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0  # Safe margin

    # Precompute conflict bitsets using Python integers
    # For each element x (0-indexed), conflicts[x] is a bitmask where bit y is set
    # if there exists a 4-AP containing x and y as two of its elements.
    # Actually, we need: for each x, the set of y such that {x, y} is part of some 4-AP.
    # But more precisely, for the greedy "can add" check, we need:
    # x can be added iff for every 4-AP containing x, at least one of the other 3 is NOT in S.
    # Equivalently: x cannot be added iff there exists a 4-AP (a, a+d, a+2d, a+3d) containing x
    # such that all other 3 elements are in S.
    #
    # Let's use a different bitset approach:
    # For each x, store the list of "conflict triples" - for each 4-AP containing x, the other 3 elements.
    # But that's what the current code does. The bitset speedup mentioned in the knowledge
    # suggests using integer bitmasks for the set S.
    #
    # Key insight: represent S as a Python integer bitmask.
    # For each element x, precompute a list of masks, where each mask represents
    # the other 3 elements in a 4-AP containing x.
    # Then x can be added iff for NO mask in the list, (S & mask) == mask.
    # i.e., x can be added iff for all masks m: (S & m) != m.
    # Equivalently: x cannot be added iff exists m: (S & m) == m.

    # Precompute for each x (0-indexed) the list of conflict masks
    conflict_masks = [[] for _ in range(N)]
    
    for d in range(1, (N // 3) + 1):
        for a in range(0, N - 3 * d):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            # For each position, the mask of the other 3
            # Position 0: mask = (1 << ap[1]) | (1 << ap[2]) | (1 << ap[3])
            # Position 1: mask = (1 << ap[0]) | (1 << ap[2]) | (1 << ap[3])
            # Position 2: mask = (1 << ap[0]) | (1 << ap[1]) | (1 << ap[3])
            # Position 3: mask = (1 << ap[0]) | (1 << ap[1]) | (1 << ap[2])
            
            m0 = (1 << ap[1]) | (1 << ap[2]) | (1 << ap[3])
            m1 = (1 << ap[0]) | (1 << ap[2]) | (1 << ap[3])
            m2 = (1 << ap[0]) | (1 << ap[1]) | (1 << ap[3])
            m3 = (1 << ap[0]) | (1 << ap[1]) | (1 << ap[2])
            
            conflict_masks[ap[0]].append(m0)
            conflict_masks[ap[1]].append(m1)
            conflict_masks[ap[2]].append(m2)
            conflict_masks[ap[3]].append(m3)
    
    # For efficiency, convert lists to tuples
    conflict_masks = [tuple(m) for m in conflict_masks]
    
    def can_add(x, S_mask):
        """Check if x (0-indexed) can be added to set represented by S_mask (int bitmask)"""
        if (S_mask >> x) & 1:
            return False
        for m in conflict_masks[x]:
            if (S_mask & m) == m:
                return False
        return True
    
    def get_size(S_mask):
        return bin(S_mask).count('1')
    
    def greedy_construct(seed):
        rng = random.Random(seed)
        S_mask = 0
        elements = list(range(N))
        rng.shuffle(elements)
        
        for x in elements:
            if can_add(x, S_mask):
                S_mask |= (1 << x)
        
        return S_mask
    
    def greedy_construct_ordered(seed):
        """Greedy that processes elements in order of fewest conflict masks first."""
        rng = random.Random(seed)
        S_mask = 0
        # Sort by number of conflict masks (fewer = easier to add)
        elements = sorted(range(N), key=lambda x: (len(conflict_masks[x]), rng.random()))
        
        for x in elements:
            if can_add(x, S_mask):
                S_mask |= (1 << x)
        
        return S_mask
    
    def local_search(S_mask, rng, time_budget):
        S_mask = S_mask
        current_size = get_size(S_mask)
        best_size = current_size
        best_S_mask = S_mask
        
        all_elements = list(range(N))
        
        while time.time() - start_time < time_budget:
            # Randomly pick an element to remove
            if current_size == 0:
                break
            x_remove = rng.choice(all_elements)
            if not ((S_mask >> x_remove) & 1):
                continue
                
            S_mask &= ~(1 << x_remove)
            current_size -= 1
            
            # Try to add any element (greedy fill)
            candidates = all_elements[:]
            rng.shuffle(candidates)
            
            for x in candidates:
                if can_add(x, S_mask):
                    S_mask |= (1 << x)
                    current_size += 1
            
            if current_size > best_size:
                best_size = current_size
                best_S_mask = S_mask
        
        # Final fill pass: try to add any remaining valid elements
        for x in all_elements:
            if can_add(x, best_S_mask):
                best_S_mask |= (1 << x)
                best_size += 1
        
        return best_S_mask
    
    # Main
    best_S_mask = 0
    best_size = 0
    
    rng = random.Random(42)
    
    # Initial greedy (random order)
    S_mask = greedy_construct(0)
    size = get_size(S_mask)
    if size > best_size:
        best_size = size
        best_S_mask = S_mask
    
    # Initial greedy (conflict-degree ordered)
    if time.time() - start_time < time_limit - 5:
        S_mask = greedy_construct_ordered(0)
        size = get_size(S_mask)
        if size > best_size:
            best_size = size
            best_S_mask = S_mask
    
    # Local search on best so far
    if time.time() - start_time < time_limit - 5:
        S_mask = local_search(best_S_mask, rng, time_limit - 5)
        size = get_size(S_mask)
        if size > best_size:
            best_size = size
            best_S_mask = S_mask
            
    # More restarts with both strategies
    seed = 1
    while time.time() - start_time < time_limit - 2:
        # Alternate between random and ordered greedy
        if seed % 2 == 0:
            S_mask = greedy_construct(seed)
        else:
            S_mask = greedy_construct_ordered(seed)
        size = get_size(S_mask)
        
        # Only do local search if promising or if we have time
        if size >= best_size - 10:
            S_mask = local_search(S_mask, rng, time_limit - 2)
            size = get_size(S_mask)
        
        if size > best_size:
            best_size = size
            best_S_mask = S_mask
        
        seed += 1
        
    # Convert bitmask to list of 1-indexed integers
    result = [i + 1 for i in range(N) if (best_S_mask >> i) & 1]
    return result

# EVOLVE-BLOCK-END
