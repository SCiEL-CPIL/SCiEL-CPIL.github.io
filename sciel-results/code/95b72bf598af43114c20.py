"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 512
    deadline = time.time() + 55.0  # Safety margin for 60s limit
    
    # Precompute for each x the list of (d, other_elements) that would form a 4-AP
    # If x is at position i in a 4-AP (a, a+d, a+2d, a+3d):
    #   i=0: a=x, need x+d, x+2d, x+3d in set
    #   i=1: a+d=x, need a=x-d, x+2d, x+3d in set -> x-d, x+d, x+2d
    #   i=2: a+2d=x, need a=x-2d, x-d, x+d in set
    #   i=3: a+3d=x, need x-3d, x-2d, x-d in set
    
    # Precompute all 4-APs for faster checking
    all_4aps = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            if a + 3*d <= n:
                all_4aps.append((a, a+d, a+2*d, a+3*d))
    
    # For each element, precompute which 4-APs it's part of and the other elements
    # element_to_aps[x] = list of (other1, other2, other3, ap_index) where x is one of the 4 elements
    element_to_aps = {x: [] for x in range(1, n + 1)}
    for idx, (a, b, c, d_val) in enumerate(all_4aps):
        for elem in (a, b, c, d_val):
            others = [x for x in (a, b, c, d_val) if x != elem]
            element_to_aps[elem].append((others[0], others[1], others[2], idx))
    
    # For speed, use a boolean array
    def can_add_fast(x, present):
        """present is a numpy boolean array of size n+1 (1-indexed)"""
        for (o1, o2, o3, _) in element_to_aps[x]:
            if present[o1] and present[o2] and present[o3]:
                return False
        return True
    
    def greedy_construct(order, present=None):
        """Greedy construction given an ordering of candidates."""
        if present is None:
            present = np.zeros(n + 1, dtype=bool)
        selected = set()
        for x in order:
            if not present[x] and can_add_fast(x, present):
                present[x] = True
                selected.add(x)
        return selected, present
    
    def local_search(selected, present, max_iters=100):
        """Try to improve selected by adding/removing elements."""
        current_size = len(selected)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding any element that can be added
            for x in range(1, n + 1):
                if not present[x] and can_add_fast(x, present):
                    present[x] = True
                    selected.add(x)
                    current_size += 1
                    improved = True
                    break
            
            if not improved:
                # Try removing one element to allow adding two
                candidates_to_remove = list(selected)
                random.shuffle(candidates_to_remove)
                found_improvement = False
                for x in candidates_to_remove[:15]:  # Try up to 15 removals
                    present[x] = False
                    selected.discard(x)
                    trial_size = current_size - 1
                    # Try to add elements
                    added = 0
                    added_elements = []
                    for y in range(1, n + 1):
                        if not present[y] and can_add_fast(y, present):
                            present[y] = True
                            added_elements.append(y)
                            added += 1
                    if added > 1:
                        for y in added_elements:
                            selected.add(y)
                        current_size = trial_size + added
                        improved = True
                        found_improvement = True
                        break
                    else:
                        # Undo the addition
                        for y in added_elements:
                            present[y] = False
                        # Put x back
                        present[x] = True
                        selected.add(x)
                        current_size += 1
                
                if not found_improvement and not improved:
                    break
        
        return selected, present
    
    # Phase 1: Multiple randomized greedy passes
    best_set = set()
    best_present = np.zeros(n + 1, dtype=bool)
    best_size = 0
    
    # Deterministic baseline
    selected, present = greedy_construct(range(1, n + 1))
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = set(selected)
        best_present = present.copy()
    
    # Randomized greedy
    candidates = list(range(1, n + 1))
    
    while time.time() < deadline - 10:  # Reserve 10s for local search
        random.shuffle(candidates)
        selected, present = greedy_construct(candidates)
        
        # Local search refinement
        selected, present = local_search(selected, present, max_iters=30)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            best_present = present.copy()
    
    # Phase 2: More aggressive local search on best found
    if time.time() < deadline - 2:
        improved_set, improved_present = local_search(best_set, best_present, max_iters=200)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
            best_present = improved_present
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
