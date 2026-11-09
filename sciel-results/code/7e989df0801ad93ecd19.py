"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 12, 'm': 40}

def _count_pairs_bitmask(masks):
    """Count unordered pairs where one mask is a strict subset of another."""
    n = len(masks)
    count = 0
    for i in range(n):
        mi = masks[i]
        for j in range(i + 1, n):
            mj = masks[j]
            # Check if mi is strict subset of mj
            if (mi & mj) == mi and mi != mj:
                count += 1
            # Check if mj is strict subset of mi
            elif (mi & mj) == mj and mi != mj:
                count += 1
    return count

def _generate_initial_chain():
    """Generate a good initial solution using nested chains."""
    n = P['n']
    m = P['m']
    
    # Start with empty set and build up chains
    # We want 40 distinct subsets from 12 vertices
    # A good strategy: create multiple chains that share some elements
    
    # Create a base chain: {}, {0}, {0,1}, {0,1,2}, ..., {0,1,...,11}
    # That's 13 sets. We need 40 total.
    
    # Strategy: Use multiple overlapping chains
    # Chain 1: sizes 0-11 (12 sets)
    # Chain 2: different elements, sizes 0-11 (12 sets) - but must be distinct
    # Actually, let's think more carefully.
    
    # Better: Use a systematic approach
    # Create sets of various sizes with good subset relationships
    
    masks = []
    used = set()
    
    # Add empty set
    masks.append(0)
    used.add(0)
    
    # Add singletons
    for i in range(n):
        mask = 1 << i
        if mask not in used:
            masks.append(mask)
            used.add(mask)
    
    # Add pairs
    for i in range(n):
        for j in range(i + 1, n):
            mask = (1 << i) | (1 << j)
            if mask not in used and len(masks) < m:
                masks.append(mask)
                used.add(mask)
            if len(masks) >= m:
                break
        if len(masks) >= m:
            break
    
    # Fill remaining with random distinct masks
    while len(masks) < m:
        mask = random.randint(1, (1 << n) - 1)
        if mask not in used:
            masks.append(mask)
            used.add(mask)
    
    return masks[:m]

def _local_search_bitmask(masks, time_limit):
    """Local search using bitmask operations for speed."""
    n = P['n']
    m = P['m']
    best_score = _count_pairs_bitmask(masks)
    current_masks = list(masks)
    used = set(current_masks)
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
        improved = False
        # Try flipping one bit in a random mask
        idx = random.randint(0, m - 1)
        old_mask = current_masks[idx]
        
        # Try all possible single-bit flips
        for bit in range(n):
            new_mask = old_mask ^ (1 << bit)
            if new_mask in used:
                continue
            
            # Try this move
            current_masks[idx] = new_mask
            used.discard(old_mask)
            used.add(new_mask)
            
            new_score = _count_pairs_bitmask(current_masks)
            
            if new_score > best_score:
                best_score = new_score
                improved = True
                no_improve_count = 0
            else:
                # Revert
                current_masks[idx] = old_mask
                used.discard(new_mask)
                used.add(old_mask)
        
        if not improved:
            no_improve_count += 1
            # Occasionally try a more aggressive move: swap two masks
            if no_improve_count % 50 == 0:
                idx1 = random.randint(0, m - 1)
                idx2 = random.randint(0, m - 1)
                if idx1 != idx2:
                    # Try replacing one mask with a random new one
                    new_mask = random.randint(0, (1 << n) - 1)
                    if new_mask not in used:
                        old_mask = current_masks[idx1]
                        current_masks[idx1] = new_mask
                        used.discard(old_mask)
                        used.add(new_mask)
                        
                        new_score = _count_pairs_bitmask(current_masks)
                        if new_score > best_score:
                            best_score = new_score
                            no_improve_count = 0
                        else:
                            current_masks[idx1] = old_mask
                            used.discard(new_mask)
                            used.add(old_mask)
    
    return current_masks, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    
    n = P['n']
    m = P['m']
    
    # Generate initial solution
    best_masks = _generate_initial_chain()
    best_score = _count_pairs_bitmask(best_masks)
    
    # Run local search
    remaining_time = deadline - time.time()
    if remaining_time > 1.0:
        best_masks, best_score = _local_search_bitmask(best_masks, remaining_time - 0.5)
    
    # Convert masks to lists of distinct integers
    result = []
    for mask in best_masks:
        lst = [b for b in range(n) if (mask >> b) & 1]
        result.append(lst)
    
    return result
# EVOLVE-BLOCK-END
