import time
import random
import numpy as np

def construct():
    n = 512
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute 4-AP conflicts
    conflicts = [[] for _ in range(n + 1)]
    
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            conflicts[a].append((b, c, e))
            conflicts[b].append((a, c, e))
            conflicts[c].append((a, b, e))
            conflicts[e].append((a, b, c))

    # Base-3 construction
    base3_set = set()
    for mask in range(1 << 6):
        val = 0
        for i in range(6):
            if (mask >> i) & 1:
                val += (3 ** i)
        if 1 <= val <= n:
            base3_set.add(val)
    
    current_mask = np.zeros(n + 1, dtype=bool)
    for x in base3_set:
        current_mask[x] = True
        
    best_mask = current_mask.copy()
    best_size = current_mask.sum()

    def can_add(mask, x):
        if mask[x]:
            return False
        for a, b, c in conflicts[x]:
            if mask[a] and mask[b] and mask[c]:
                return False
        return True

    def greedy_add(mask):
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if not mask[x] and can_add(mask, x):
                    mask[x] = True
                    changed = True
        return mask

    # Initial greedy pass
    current_mask = greedy_add(current_mask)
    best_size = current_mask.sum()
    best_mask = current_mask.copy()

    random.seed(42)
    
    # Local Search
    # We will alternate between greedy add and "Remove 1, Add 1 or 2" moves.
    
    while time.time() < deadline:
        # 1. Greedy Add
        current_mask = greedy_add(current_mask)
        current_size = current_mask.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask.copy()
            
        # Check if we are stuck (no single element can be added)
        stuck = True
        for x in range(1, n + 1):
            if not current_mask[x] and can_add(current_mask, x):
                stuck = False
                break
                
        if not stuck:
            continue # Greedy will handle it in next iteration
            
        # 2. If stuck, try a move: Remove u, Add v (and maybe w)
        in_set = [x for x in range(1, n + 1) if current_mask[x]]
        not_in_set = [x for x in range(1, n + 1) if not current_mask[x]]
        
        if not in_set or not not_in_set:
            break
            
        u = random.choice(in_set)
        
        # Remove u
        current_mask[u] = False
        
        # Try to add v
        v = random.choice(not_in_set)
        if can_add(current_mask, v):
            current_mask[v] = True
            # Try to add w
            # Pick a random w from not_in_set (excluding v)
            # To save time, just pick random from not_in_set and check if it's v
            w = random.choice(not_in_set)
            if w != v and can_add(current_mask, w):
                current_mask[w] = True
                
            # After additions, try greedy add to fill any new gaps
            current_mask = greedy_add(current_mask)
            current_size = current_mask.sum()
            
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask.copy()
        else:
            # Revert
            current_mask[u] = True

    return sorted(best_mask.nonzero()[0].tolist())
