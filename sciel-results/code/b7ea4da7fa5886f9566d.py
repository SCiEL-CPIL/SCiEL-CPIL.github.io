import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 9
    # Generate K_9^(5) as bitmasks
    # Vertex i is bit (1 << i)
    current_edges_masks = []
    for e in combinations(range(n), 5):
        mask = 0
        for v in e:
            mask |= (1 << v)
        current_edges_masks.append(mask)
    
    # Popcount helper
    # Python 3.8+ has int.bit_count()
    
    def is_2_colorable(edges_masks, n):
        # mask0: vertices colored 0
        # mask1: vertices colored 1
        # unassigned: ~(mask0 | mask1)
        
        mask0 = 0
        mask1 = 0
        
        def propagate():
            changed = True
            while changed:
                changed = False
                for e in edges_masks:
                    c0 = (e & mask0).bit_count()
                    c1 = (e & mask1).bit_count()
                    unassigned = e & ~(mask0 | mask1)
                    
                    if c0 == 5 or c1 == 5:
                        return False
                    
                    if c0 == 4 and unassigned:
                        # Find the unassigned vertex
                        # It must be the only one? No, if c0=4, there is 1 unassigned.
                        v_bit = unassigned
                        # v_bit is a power of 2
                        if v_bit & mask0:
                            return False
                        if v_bit & mask1:
                            return False
                        mask1 |= v_bit
                        changed = True
                        
                    if c1 == 4 and unassigned:
                        v_bit = unassigned
                        if v_bit & mask0:
                            return False
                        if v_bit & mask1:
                            return False
                        mask0 |= v_bit
                        changed = True
            return True

        def backtrack(idx):
            if idx == n:
                return True
            
            bit = 1 << idx
            
            # Try 0
            if not (mask1 & bit): # Not already 1
                orig0 = mask0
                orig1 = mask1
                mask0 |= bit
                if propagate():
                    if backtrack(idx + 1):
                        return True
                mask0 = orig0
                mask1 = orig1

            # Try 1
            if not (mask0 & bit): # Not already 0
                orig0 = mask0
                orig1 = mask1
                mask1 |= bit
                if propagate():
                    if backtrack(idx + 1):
                        return True
                mask0 = orig0
                mask1 = orig1
                
            return False

        if not propagate():
            return False
            
        # Find first unassigned
        first = -1
        for i in range(n):
            bit = 1 << i
            if not (mask0 | mask1) & bit:
                first = i
                break
                
        if first == -1:
            return True
            
        return backtrack(first)

    if is_2_colorable(current_edges_masks, n):
        return {'n': n, 'edges': [list(e) for e in combinations(range(n), 5)]}

    best_masks = current_edges_masks[:]
    best_count = len(best_masks)
    
    while time.time() < deadline:
        if not current_edges_masks:
            break
            
        idx = random.randint(0, len(current_edges_masks) - 1)
        edge_to_remove = current_edges_masks[idx]
        
        new_masks = [m for i, m in enumerate(current_edges_masks) if i != idx]
        
        if not is_2_colorable(new_masks, n):
            current_edges_masks = new_masks
            if len(current_edges_masks) < best_count:
                best_count = len(current_edges_masks)
                best_masks = current_edges_masks[:]
                
    # Convert best_masks back to list of lists
    best_edges_list = []
    for m in best_masks:
        edge = []
        for i in range(n):
            if m & (1 << i):
                edge.append(i)
        best_edges_list.append(edge)
        
    return {'n': n, 'edges': best_edges_list}
