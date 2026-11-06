"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit
    
    # Use bitmask for subset sums: bit i set means sum i is reachable
    # Bit 0 is always set (sum 0)
    
    def greedy(order, deadline):
        """Greedy construction given an order of elements to consider."""
        selected = []
        mask = 1  # bit 0 set, sum 0 reachable
        mask_limit = (1 << 201) - 1  # bits 0..200
        
        for x in order:
            if time.time() > deadline:
                break
            # Check if adding x would create sum 200
            # We need: is (200 - x) reachable? i.e., bit (200-x) set in mask
            if x <= 200:
                if mask & (1 << (200 - x)):
                    continue
            # Add x: shift mask left by x, OR with old mask
            new_mask = (mask << x) | mask
            # Only keep bits 0..200
            new_mask &= mask_limit
            mask = new_mask
            selected.append(x)
        
        return selected, mask
    
    def try_add(mask, x):
        """Check if we can add x to a set with given reachable mask."""
        if x > 200:
            return False
        if mask & (1 << (200 - x)):
            return False
        return True
    
    def add_element(mask, x):
        """Add x to the set, return new mask."""
        new_mask = (mask << x) | mask
        return new_mask & ((1 << 201) - 1)
    
    best = None
    best_len = 0
    best_mask = 1
    
    # Try multiple orderings
    orderings = []
    orderings.append(list(range(100, 0, -1)))
    orderings.append(list(range(1, 101)))
    
    rng = random.Random(42)
    for _ in range(150):
        perm = list(range(1, 101))
        rng.shuffle(perm)
        orderings.append(perm)
    
    for order in orderings:
        if time.time() > deadline:
            break
        result, mask = greedy(order, deadline)
        if len(result) > best_len:
            best_len = len(result)
            best = result
            best_mask = mask
    
    if best is None or len(best) == 0:
        return [100]
    
    # Local improvement: try to add excluded elements
    best_set = set(best)
    excluded = [x for x in range(1, 101) if x not in best_set]
    improved = list(best)
    mask = best_mask
    
    # Try adding elements that can be added
    for x in excluded:
        if time.time() > deadline:
            break
        if try_add(mask, x):
            mask = add_element(mask, x)
            improved.append(x)
            best_set.add(x)
    
    # Local improvement: try swaps - remove one element, then try to add multiple excluded
    if time.time() < deadline:
        best_improved_len = len(improved)
        improved_set = set(improved)
        
        for remove_x in list(improved):
            if time.time() > deadline:
                break
            # Remove remove_x from the set
            # We need to recompute the mask without remove_x
            # This is tricky with bitmask; let's just try: remove and see what we can add back
            temp_improved = [y for y in improved if y != remove_x]
            temp_set = set(temp_improved)
            temp_mask = 1
            for y in temp_improved:
                temp_mask = add_element(temp_mask, y)
            
            # Now try to add all excluded elements (including remove_x and others)
            for x in range(1, 101):
                if time.time() > deadline:
                    break
                if x not in temp_set and try_add(temp_mask, x):
                    temp_mask = add_element(temp_mask, x)
                    temp_set.add(x)
            
            if len(temp_set) > best_improved_len:
                best_improved_len = len(temp_set)
                improved = list(temp_set)
                mask = temp_mask
                improved_set = temp_set
    
    if len(improved) > best_len:
        best = improved
        best_len = len(improved)
    
    return best
# EVOLVE-BLOCK-END
