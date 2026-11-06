"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 256
    # Precompute conflict graph: conflict[i][j] = True if (i*j) % (i+j) == 0
    # We want to find a large subset with NO conflicts (independent set).
    # Build adjacency as bitmasks for speed.
    
    # conflict_mask[i] is a bitmask where bit j is set if i and j conflict
    conflict_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Also store for quick lookup
    # Note: we only need to check distinct pairs, so no self-conflicts
    
    def popcount(x):
        return bin(x).count('1')
    
    def evaluate(selection):
        """Check if selection is valid. selection is a bitmask."""
        # For each pair in selection, check no conflict
        # This is expensive, so we use it sparingly
        # Actually, we maintain validity incrementally in search
        pass
    
    # Greedy initial solution: start with all, remove conflicts
    # Better: greedy construction
    # Start with all nodes, iteratively remove the node with most conflicts in remaining set
    
    # Alternative: start with a large subset and do local search
    # Let's start with all odd numbers (known to work well) and try to add evens
    
    # Actually, let's do a proper local search from a good initial solution
    # Initial: all numbers 1..256, then remove conflicting nodes greedily
    
    # Greedy: repeatedly pick the node with smallest degree in remaining conflict graph
    # But that's for finding independent set. Let's use a different approach.
    
    # Start with all nodes, then remove nodes to resolve conflicts
    # This gives a maximal independent set
    
    # Better approach: start with a large valid set and try to add/remove/swap
    
    # Let's use the odd numbers as a strong starting point
    # All odd numbers: check if any two odds conflict
    # (a*b) % (a+b) == 0 for odd a,b
    # a+b is even, a*b is odd, so odd % even != 0 always? 
    # No: odd % even can be 0 only if even divides odd, which means even <= odd and even | odd.
    # Since even > 1 and odd, the only way is if a+b divides a*b.
    # For odd a,b: a+b is even, a*b is odd. An even number cannot divide an odd number.
    # So NO two odd numbers conflict! Great, all odds are a valid set.
    
    # So start with all 128 odd numbers.
    initial = set(range(1, N + 1, 2))
    current_mask = 0
    for x in initial:
        current_mask |= (1 << x)
    
    best_mask = current_mask
    best_size = len(initial)
    
    # Try to add even numbers one by one
    evens = list(range(2, N + 1, 2))
    random.shuffle(evens)
    
    def can_add(mask, x):
        """Check if x can be added to the set represented by mask."""
        # x must not conflict with any element in mask
        # conflict_mask[x] has bits set for all y that conflict with x
        # We need (conflict_mask[x] & mask) == 0
        return (conflict_mask[x] & mask) == 0
    
    def remove_node(mask, x):
        return mask & ~(1 << x)
    
    # Greedy: try to add each even number if it doesn't conflict
    for x in evens:
        if can_add(current_mask, x):
            current_mask |= (1 << x)
    
    if popcount(current_mask) > best_size:
        best_mask = current_mask
        best_size = popcount(current_mask)
    
    # Local search: try swaps (remove one, add another) and additions
    # Also try removing nodes to allow more additions
    
    current = current_mask
    current_size = popcount(current)
    
    # Phase 1: try to add any remaining node
    changed = True
    while changed and (time.time() - start_time) < DEADLINE - 10:
        changed = False
        for x in range(1, N + 1):
            if (current >> x) & 1:
                continue
            if can_add(current, x):
                current |= (1 << x)
                current_size += 1
                changed = True
                if current_size > best_size:
                    best_mask = current
                    best_size = current_size
    
    # Phase 2: swap moves - remove one node, try to add two others
    # Or remove one, add one (net 0 or net +1 if we can add more after)
    
    # Better: try all pairs (remove a, add b) where b not in set, a in set
    # This is O(n^2) per iteration, which is 256^2 = 65536, manageable
    
    no_improve_count = 0
    max_no_improve = 50
    
    while (time.time() - start_time) < DEADLINE - 5 and no_improve_count < max_no_improve:
        improved = False
        # Try single additions first
        for x in range(1, N + 1):
            if (current >> x) & 1:
                continue
            if can_add(current, x):
                current |= (1 << x)
                current_size += 1
                improved = True
                if current_size > best_size:
                    best_mask = current
                    best_size = current_size
        
        if not improved:
            # Try swaps: remove a, add b
            # To be efficient, for each b not in set, find a in set such that
            # removing a allows b to be added
            # i.e., conflict_mask[b] & (current without a) == 0
            # which means conflict_mask[b] & current must only have bit a set
            # i.e., (conflict_mask[b] & current) has popcount <= 1
            # and if popcount == 1, that bit is a
            for b in range(1, N + 1):
                if (current >> b) & 1:
                    continue
                cm = conflict_mask[b] & current
                pc = popcount(cm)
                if pc == 0:
                    # Can add b directly
                    current |= (1 << b)
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_mask = current
                        best_size = current_size
                elif pc == 1:
                    # Find the conflicting node a
                    # a is the only bit set in cm
                    a = cm.bit_length() - 1
                    # Remove a, add b: net change 0 in size
                    # But maybe after removing a, we can add more?
                    # For now, do the swap only if it enables further additions
                    # Let's try: remove a, add b, then try to add more
                    new_mask = (current & ~(1 << a)) | (1 << b)
                    new_size = current_size  # same size
                    # Try to add more to new_mask
                    can_improve = False
                    for c in range(1, N + 1):
                        if (new_mask >> c) & 1:
                            continue
                        if can_add(new_mask, c):
                            can_improve = True
                            break
                    if can_improve:
                        current = new_mask
                        improved = True
                        # Don't update best yet, wait for additions
                        # Actually, let's just do the swap and let the next iteration add
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
    
    # Phase 3: try 2-removal, 3-addition moves (more complex)
    # This is expensive, so only do a limited number of iterations
    
    if (time.time() - start_time) < DEADLINE - 3:
        # Try removing pairs and adding triples
        # This is O(n^2 * n) which is too slow, so sample
        for _ in range(1000):
            if (time.time() - start_time) > DEADLINE - 3:
                break
            # Pick two random nodes to remove
            nodes_in_set = [x for x in range(1, N + 1) if (current >> x) & 1]
            if len(nodes_in_set) < 2:
                break
            a, b = random.sample(nodes_in_set, 2)
            new_mask = current & ~(1 << a) & ~(1 << b)
            new_size = current_size - 2
            # Try to add nodes
            added = 0
            for c in range(1, N + 1):
                if (new_mask >> c) & 1:
                    continue
                if can_add(new_mask, c):
                    new_mask |= (1 << c)
                    new_size += 1
                    added += 1
                    if added >= 3:
                        break
            if new_size > current_size:
                current = new_mask
                current_size = new_size
                if current_size > best_size:
                    best_mask = current
                    best_size = current_size
    
    # Final: try single additions again
    for x in range(1, N + 1):
        if (current >> x) & 1:
            continue
        if can_add(current, x):
            current |= (1 << x)
            current_size += 1
            if current_size > best_size:
                best_mask = current
                best_size = current_size
    
    # Return the best set found
    result = []
    for x in range(1, N + 1):
        if (best_mask >> x) & 1:
            result.append(x)
    
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
