import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    N = 128
    
    # 1. Precompute GCDs
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # 2. Precompute Conflict Lists
    # Conflict[x] is a list of pairs (a, b) such that (x, a, b) is a bad triple.
    conflict = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                g_ab = gcd_mat[a][b]
                g_ac = gcd_mat[a][c]
                g_bc = gcd_mat[b][c]
                if g_ab == g_ac and g_ab == g_bc:
                    # Triple (a, b, c) is bad
                    conflict[a].append((b, c))
                    conflict[b].append((a, c))
                    conflict[c].append((a, b))
                    
    def can_add(x, in_set):
        for a, b in conflict[x]:
            if in_set[a] and in_set[b]:
                return False
        return True

    def greedy_fill(in_set, mask):
        """Try to add any element that can be added. Returns new mask, in_set, and count of added elements."""
        added = 0
        # We loop until no more can be added? Or just one pass?
        # One pass might miss elements that become addable after another is added?
        # No, adding an element only increases conflicts, so if x can't be added now, it can't be added later.
        # So one pass is sufficient for a greedy fill from a given state.
        for x in range(1, N + 1):
            if not in_set[x]:
                if can_add(x, in_set):
                    in_set[x] = True
                    mask |= (1 << x)
                    added += 1
        return mask, in_set, added

    def local_search(mask, in_set):
        best_mask = mask
        best_size = bin(mask).count('1')
        
        while time.time() < deadline:
            improved = False
            
            # 1. Try to add any element
            new_mask, new_in_set, added = greedy_fill(in_set[:], mask)
            if added > 0:
                in_set = new_in_set
                mask = new_mask
                if bin(mask).count('1') > best_size:
                    best_size = bin(mask).count('1')
                    best_mask = mask
                improved = True
                continue # Restart loop to try more additions

            # 2. If stuck, perform kicks
            current_elements = [i for i in range(1, N + 1) if in_set[i]]
            if not current_elements:
                break

            found_better = False
            
            # Try 1-element kicks
            for _ in range(20): # Increased kicks
                if time.time() > deadline:
                    break
                u = random.choice(current_elements)
                
                # Save state
                old_in_set = in_set[:]
                old_mask = mask
                
                # Remove u
                in_set[u] = False
                mask &= ~(1 << u)
                
                # Greedy fill
                new_mask, new_in_set, added = greedy_fill(in_set, mask)
                
                # Check if improved
                new_size = bin(new_mask).count('1')
                if new_size > best_size:
                    best_mask = new_mask
                    best_size = new_size
                    in_set = new_in_set
                    mask = new_mask
                    found_better = True
                    break
                else:
                    # Revert
                    in_set = old_in_set
                    mask = old_mask
            
            if found_better:
                continue

            # Try 2-element kicks if 1-element failed
            if len(current_elements) >= 2:
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    u, v = random.sample(current_elements, 2)
                    
                    # Save state
                    old_in_set = in_set[:]
                    old_mask = mask
                    
                    # Remove u, v
                    in_set[u] = False
                    in_set[v] = False
                    mask &= ~(1 << u)
                    mask &= ~(1 << v)
                    
                    # Greedy fill
                    new_mask, new_in_set, added = greedy_fill(in_set, mask)
                    
                    # Check if improved
                    new_size = bin(new_mask).count('1')
                    if new_size > best_size:
                        best_mask = new_mask
                        best_size = new_size
                        in_set = new_in_set
                        mask = new_mask
                        found_better = True
                        break
                    else:
                        # Revert
                        in_set = old_in_set
                        mask = old_mask
            
            if found_better:
                continue

            # If neither 1 nor 2 kicks improved, we might be stuck.
            # Break to allow outer loop to restart with different init.
            break
            
        return best_mask

    best_mask = 0
    best_size = 0
    
    # Strategy 1: Natural Order
    in_set = [False] * (N + 1)
    mask = 0
    # Initial greedy
    for x in range(1, N + 1):
        if can_add(x, in_set):
            in_set[x] = True
            mask |= (1 << x)
            
    m = local_search(mask, in_set)
    if bin(m).count('1') > best_size:
        best_mask = m
        best_size = bin(m).count('1')
        
    # Strategy 2: Reverse Order
    if time.time() < deadline:
        in_set = [False] * (N + 1)
        mask = 0
        for x in range(N, 0, -1):
            if can_add(x, in_set):
                in_set[x] = True
                mask |= (1 << x)
        m = local_search(mask, in_set)
        if bin(m).count('1') > best_size:
            best_mask = m
            best_size = bin(m).count('1')

    # Strategy 3: Random Restarts
    while time.time() < deadline - 2: # Keep 2s buffer
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        in_set = [False] * (N + 1)
        mask = 0
        for x in random_order:
            if can_add(x, in_set):
                in_set[x] = True
                mask |= (1 << x)
        m = local_search(mask, in_set)
        if bin(m).count('1') > best_size:
            best_mask = m
            best_size = bin(m).count('1')
            
    # Convert mask to list
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << i):
            result.append(i)
            
    if not result:
        result = [1]
        
    return result
