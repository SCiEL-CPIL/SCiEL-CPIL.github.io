# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.time() + 620  # safety margin before 630s kill

    # Precompute forbidden set
    forbidden = set()
    for x in range(1, m):
        v = (x * x) % m
        if v != 0:
            forbidden.add(v)

    # Precompute conflict bitmasks
    # conflict_mask[i] is a bitmask of vertices conflicting with i
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            if d1 in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask

    def can_add(v, selected_mask):
        """Check if v can be added to selected_mask."""
        return (conflict_mask[v] & selected_mask) == 0

    def greedy_init(rng):
        """Greedy: add elements in random order if they don't conflict with already selected."""
        order = list(range(m))
        rng.shuffle(order)
        selected_mask = 0
        for v in order:
            if can_add(v, selected_mask):
                selected_mask |= (1 << v)
        return selected_mask

    def local_search(selected_mask, rng, deadline):
        """
        Local search using bitmasks.
        """
        # Convert mask to list for iteration
        selected_list = [i for i in range(m) if (selected_mask >> i) & 1]
        best_size = len(selected_list)
        no_improve_count = 0
        
        while time.time() < deadline - 5:
            improved = False
            current_size = len(selected_list)
            
            # 1. Try adding a vertex
            # Find candidates that don't conflict with any selected
            # We can iterate all v not in selected
            for v in range(m):
                if (selected_mask >> v) & 1:
                    continue
                if can_add(v, selected_mask):
                    selected_mask |= (1 << v)
                    selected_list.append(v)
                    improved = True
                    break
            
            if not improved:
                # 2. Try 1-for-1 swap
                # Remove a, add b
                for a in selected_list[:]:  # iterate over copy
                    mask_without_a = selected_mask & ~(1 << a)
                    for b in range(m):
                        if (selected_mask >> b) & 1:
                            continue
                        if can_add(b, mask_without_a):
                            # Check if b conflicts with a? No, a is removed.
                            # But we need to ensure b doesn't conflict with others in mask_without_a.
                            # can_add does that.
                            # Also need to ensure a and b don't conflict? No, a is removed.
                            selected_mask = mask_without_a | (1 << b)
                            # Update list
                            selected_list.remove(a)
                            selected_list.append(b)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                # 3. Try 1-for-2 swap
                # Remove a, add b, c
                for a in selected_list[:]:
                    mask_without_a = selected_mask & ~(1 << a)
                    found = False
                    for b in range(m):
                        if (selected_mask >> b) & 1:
                            continue
                        if not can_add(b, mask_without_a):
                            continue
                        # Now try to add c to (mask_without_a | (1<<b))
                        mask_with_b = mask_without_a | (1 << b)
                        for c in range(m):
                            if (selected_mask >> c) & 1 or c == b:
                                continue
                            if can_add(c, mask_with_b):
                                # Also need c not conflict with b
                                # can_add(c, mask_with_b) checks c against all in mask_with_b, which includes b.
                                selected_mask = mask_with_b | (1 << c)
                                selected_list.remove(a)
                                selected_list.append(b)
                                selected_list.append(c)
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
            
            if improved:
                no_improve_count = 0
                if len(selected_list) > best_size:
                    best_size = len(selected_list)
            else:
                no_improve_count += 1
                # Perturb
                if no_improve_count >= 20 and selected_list:
                    to_remove = rng.sample(selected_list, min(3, len(selected_list)))
                    for r in to_remove:
                        selected_mask &= ~(1 << r)
                        selected_list.remove(r)
                    no_improve_count = 0

        return selected_mask

    # Multi-start local search
    best_mask = 0
    best_size = 0
    rng = random.Random(42)
    
    # First, try a few greedy inits
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        sel_mask = greedy_init(rng)
        if bin(sel_mask).count('1') > best_size:
            best_mask = sel_mask
            best_size = bin(sel_mask).count('1')
        
        # Do local search on this
        # Pass a copy or let local_search handle it? 
        # local_search modifies local variables.
        sel_mask_optimized = local_search(sel_mask, rng, deadline)
        cur_size = bin(sel_mask_optimized).count('1')
        if cur_size > best_size:
            best_mask = sel_mask_optimized
            best_size = cur_size
    
    # Continue local search on best found
    if time.time() < deadline - 5 and best_mask != 0:
        best_mask = local_search(best_mask, rng, deadline)
        best_size = bin(best_mask).count('1')
    
    # Convert best_mask to array
    result = np.array([i for i in range(m) if (best_mask >> i) & 1], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
