import time
import random
from itertools import combinations

def is_2_colorable(n, edges, timeout=None):
    """
    Check if hypergraph is 2-colorable using backtracking with unit propagation.
    Uses bitmasks for speed.
    Returns True if 2-colorable, False if not.
    """
    if timeout is not None:
        start = time.time()
    
    # Precompute edge masks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # assigned: bitmask of assigned vertices
    # color1: bitmask of vertices colored 1 (subset of assigned)
    assigned = 0
    color1 = 0
    
    def unit_propagate():
        nonlocal assigned, color1
        changed = True
        while changed:
            changed = False
            for em in edge_masks:
                unassigned = em & ~assigned
                # If no unassigned, check for monochromatic
                if unassigned == 0:
                    # All assigned
                    if (em & color1) == em: # All 1
                        return False
                    if (em & color1) == 0: # All 0
                        return False
                elif (unassigned & (unassigned - 1)) == 0:
                    # Exactly one unassigned
                    v_bit = unassigned
                    # Count assigned 1s and 0s in this edge
                    assigned_part = em & assigned
                    c1_count = bin(assigned_part & color1).count('1')
                    c0_count = bin(assigned_part & ~color1).count('1')
                    
                    if c1_count == 4 and c0_count == 0:
                        # Must be 0
                        assigned |= v_bit
                        changed = True
                    elif c0_count == 4 and c1_count == 0:
                        # Must be 1
                        assigned |= v_bit
                        color1 |= v_bit
                        changed = True
        return True

    def backtrack():
        if timeout is not None:
            if time.time() - start > timeout:
                return None
        
        if assigned == (1 << n) - 1:
            return True
            
        unassigned_mask = ~assigned & ((1 << n) - 1)
        if unassigned_mask == 0:
             return True
        
        # Pick a vertex. 
        # Heuristic: pick vertex in an edge with max assigned?
        # For speed, just pick lowest.
        v_bit = unassigned_mask & -unassigned_mask
        
        old_assigned = assigned
        old_color1 = color1
        
        # Try 0
        assigned = old_assigned | v_bit
        # color1 unchanged
        
        if unit_propagate():
            res = backtrack()
            if res is True:
                return True
        
        # Try 1
        assigned = old_assigned | v_bit
        color1 = old_color1 | v_bit
        
        if unit_propagate():
            res = backtrack()
            if res is True:
                return True
        
        assigned = old_assigned
        color1 = old_color1
        return None

    if not unit_propagate():
        return False
        
    result = backtrack()
    if result is True:
        return True
    elif result is False:
        return False
    else:
        return True

def construct():
    deadline = time.time() + 120
    best_n = None
    best_edges = None
    best_count = float('inf')
    
    # We know m(5) <= 51.
    # We want to find < 51.
    
    # Search n from 5 to 26
    for n in range(5, 27):
        if time.time() > deadline - 10:
            break
            
        all_edges = list(combinations(range(n), 5))
        num_possible = len(all_edges)
        
        if num_possible < 5:
            continue
            
        # If we already have a good solution, we only care about smaller ones
        # If best_count is 51, we want < 51.
        
        # Try random subsets
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            random.shuffle(all_edges)
            
            # Start size: 
            # If best_count is inf, start with 30.
            # If best_count is 51, start with 40? 
            # We want to find non-2-colorable, then minimize.
            
            start_size = min(num_possible, max(10, best_count - 5 if best_count < inf else 40))
            
            current_edges = all_edges[:start_size]
            
            # Check if non-2-colorable
            # Use short timeout for initial check
            if not is_2_colorable(n, current_edges, timeout=0.5):
                # It is non-2-colorable.
                # Minimize
                changed = True
                while changed:
                    changed = False
                    for i in range(len(current_edges)):
                        if time.time() > deadline - 2:
                            break
                        test_edges = current_edges[:i] + current_edges[i+1:]
                        # Check if still non-2-colorable
                        # If it is 2-colorable, we must keep edge i.
                        # If it is non-2-colorable, we can remove edge i.
                        if not is_2_colorable(n, test_edges, timeout=0.5):
                            current_edges = test_edges
                            changed = True
                            break # Restart loop
                            
                if len(current_edges) < best_count:
                    best_count = len(current_edges)
                    best_n = n
                    best_edges = current_edges
                    
                # If we found a very small one, we can stop early?
                # No, keep searching for even smaller.
                
            # If it was 2-colorable, we could try adding edges?
            # But random restart is simpler.

    # Fallback
    if best_edges is None:
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_count = len(best_edges)
        
    return {'n': best_n, 'edges': best_edges}
