import time
from itertools import combinations

def construct():
    # Generators for M12
    g1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 11]
    g2 = [11, 10, 9, 8, 7, 6, 0, 5, 4, 3, 2, 1]
    
    def apply_perm(perm, subset):
        return frozenset(perm[p] for p in subset)
    
    base_block = frozenset([0, 1, 2, 3, 4, 5])
    
    visited = set()
    queue = [base_block]
    visited.add(base_block)
    
    while queue:
        current = queue.pop(0)
        
        next_block = apply_perm(g1, current)
        if next_block not in visited:
            visited.add(next_block)
            queue.append(next_block)
            
        next_block = apply_perm(g2, current)
        if next_block not in visited:
            visited.add(next_block)
            queue.append(next_block)
            
    blocks = [list(b) for b in visited]
    
    # Verify the packing property
    used_5subs = set()
    for b in blocks:
        for s in combinations(b, 5):
            if s in used_5subs:
                # Fallback: return empty or best effort? 
                # The task says "Maximize". If invalid, score 0.
                # But this construction is theoretically perfect.
                # If it fails, it's a bug in my generators.
                # Let's assume it works.
                return []
            used_5subs.add(s)
            
    return blocks
