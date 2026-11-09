"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct a large 4-term AP-free subset of [1, 512].
    
    Strategy:
    1. Start with a Salem-Spencer set (base-3 digits 0,1 only) as a valid seed.
    2. Use greedy insertion to add elements that don't create 4-APs.
    3. Apply local search (swap/insert/delete) to maximize size.
    """
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin out of 90s limit
    
    n = 512
    rng = random.Random(42)
    
    # Precompute all 4-APs in [1, 512]
    # A 4-AP is (a, a+d, a+2d, a+3d) where a>=1, a+3d<=512, d>=1
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map each element to the APs it participates in
    elem_to_aps = {i: [] for i in range(1, n + 1)}
    for idx, ap in enumerate(ap_list):
        for elem in ap:
            elem_to_aps[elem].append(idx)
    
    def has_4ap_in_set(selected_set):
        """Check if a set contains any 4-AP."""
        for ap in ap_list:
            if all(x in selected_set for x in ap):
                return True
        return False
    
    def can_add(selected_set, elem):
        """Check if adding elem to selected_set creates a 4-AP."""
        for ap_idx in elem_to_aps[elem]:
            ap = ap_list[ap_idx]
            # Check if all other elements in this AP are already in selected_set
            if all(x in selected_set for x in ap if x != elem):
                return False
        return True
    
    # Step 1: Generate initial Salem-Spencer set
    # Numbers in [1, 512] whose base-3 representation uses only 0 and 1
    salem_spencer = set()
    for x in range(1, n + 1):
        # Convert to base 3
        temp = x
        valid = True
        while temp > 0:
            if temp % 3 == 2:
                valid = False
                break
            temp //= 3
        if valid:
            salem_spencer.add(x)
    
    # Step 2: Greedy insertion to maximize size
    selected = set(salem_spencer)
    
    # Try to add elements in random order
    candidates = [x for x in range(1, n + 1) if x not in selected]
    rng.shuffle(candidates)
    
    for elem in candidates:
        if time.time() > deadline:
            break
        if can_add(selected, elem):
            selected.add(elem)
    
    # Step 3: Local search to improve
    best_set = set(selected)
    best_size = len(best_set)
    
    # Local search: try swaps (remove one, add another) and pure insertions
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try pure insertions
        for elem in range(1, n + 1):
            if elem not in selected and can_add(selected, elem):
                selected.add(elem)
                improved = True
                break
        
        if improved:
            if len(selected) > best_size:
                best_set = set(selected)
                best_size = len(best_set)
            continue
        
        # Try swaps: remove one element, try to add two
        if len(selected) > 0:
            # Randomly pick an element to remove
            to_remove = rng.choice(list(selected))
            selected.discard(to_remove)
            
            # Try to add elements
            added = 0
            for elem in range(1, n + 1):
                if elem not in selected and can_add(selected, elem):
                    selected.add(elem)
                    added += 1
                    if added >= 2:
                        break
            
            if added >= 2:
                if len(selected) > best_size:
                    best_set = set(selected)
                    best_size = len(best_set)
                improved = True
            else:
                # Revert the swap
                selected.discard(to_remove)
                # Remove any elements we added
                for elem in range(1, n + 1):
                    if elem in selected and elem != to_remove and can_add(selected - {elem}, elem) == False:
                        pass
                # Actually, let's just revert properly
                # We need to track what we added
                # Simpler: just revert to best_set if no improvement
                selected = set(best_set)
        
        if not improved:
            # Perturb: remove a random element and try to refill
            if len(selected) > 0:
                to_remove = rng.choice(list(selected))
                selected.discard(to_remove)
                for elem in range(1, n + 1):
                    if elem not in selected and can_add(selected, elem):
                        selected.add(elem)
    
    # Return the best set found
    return sorted(best_set)
# EVOLVE-BLOCK-END
