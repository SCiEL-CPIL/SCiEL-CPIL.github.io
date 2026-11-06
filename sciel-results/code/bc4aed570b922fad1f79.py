import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0
    N = 256
    
    # Precompute conflict using bitmask
    # conflict_mask[i] has bit j set if i and j conflict
    conflict_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = math.gcd(a, b)
            m = a // d
            n = b // d
            if d % (m + n) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    
    odds = [x for x in range(1, N + 1) if x % 2 == 1]
    evens = [x for x in range(1, N + 1) if x % 2 == 0]
    
    def greedy_fill(selected_mask, selected_set, candidates):
        """Greedily add candidates that don't conflict with selected."""
        added = []
        for x in candidates:
            if x in selected_set:
                continue
            if conflict_mask[x] & selected_mask:
                continue
            selected_set.add(x)
            selected_mask |= (1 << x)
            added.append(x)
        return selected_mask, selected_set, added
    
    def try_add_all(selected_mask, selected_set):
        """Try to add any number from 1..N that doesn't conflict."""
        improved = True
        while improved:
            improved = False
            for x in range(1, N + 1):
                if x in selected_set:
                    continue
                if conflict_mask[x] & selected_mask:
                    continue
                selected_set.add(x)
                selected_mask |= (1 << x)
                improved = True
        return selected_mask, selected_set
    
    best_result = set()
    best_size = 0
    
    # Initial: all odds + greedy evens
    selected = set(odds)
    mask = 0
    for x in odds:
        mask |= (1 << x)
    mask, selected, _ = greedy_fill(mask, selected, evens)
    mask, selected = try_add_all(mask, selected)
    best_result = set(selected)
    best_size = len(selected)
    
    # Local search: try removing one element and see if we can add more
    def local_search(selected_set, selected_mask, time_limit):
        selected_set = set(selected_set)
        selected_mask = selected_mask
        best_size_ls = len(selected_set)
        best_set_ls = set(selected_set)
        best_mask_ls = selected_mask
        
        t_start = time.time()
        no_improve_count = 0
        
        while time.time() < time_limit and no_improve_count < 20:
            improved = False
            # Try removing each element
            for x in list(selected_set):
                if time.time() >= time_limit:
                    break
                # Remove x
                selected_set.remove(x)
                selected_mask ^= (1 << x)
                
                # Try to add as many as possible
                added = []
                for y in range(1, N + 1):
                    if y in selected_set or y == x:
                        continue
                    if conflict_mask[y] & selected_mask:
                        continue
                    selected_set.add(y)
                    selected_mask |= (1 << y)
                    added.append(y)
                
                new_size = len(selected_set)
                if new_size > best_size_ls:
                    best_size_ls = new_size
                    best_set_ls = set(selected_set)
                    best_mask_ls = selected_mask
                    improved = True
                    no_improve_count = 0
                else:
                    # Revert
                    for y in added:
                        selected_set.remove(y)
                        selected_mask ^= (1 << y)
                    selected_set.add(x)
                    selected_mask |= (1 << x)
            
            if not improved:
                no_improve_count += 1
        
        return best_set_ls, best_mask_ls, best_size_ls
    
    # Local search from initial
    ls_set, ls_mask, ls_size = local_search(best_result, mask, t0 + 60.0)
    if ls_size > best_size:
        best_result = ls_set
        best_size = ls_size
        mask = ls_mask
    
    # Multi-start
    random.seed(123)
    while time.time() < DEADLINE - 5.0:
        # Random perturbation: remove 1-5 random elements from best
        perturbed = set(best_result)
        num_remove = random.randint(1, 5)
        to_remove = random.sample(list(perturbed), min(num_remove, len(perturbed)))
        for x in to_remove:
            perturbed.discard(x)
        
        p_mask = 0
        for x in perturbed:
            p_mask |= (1 << x)
        
        # Greedy fill
        p_mask, perturbed, _ = greedy_fill(p_mask, perturbed, list(range(1, N + 1)))
        p_mask, perturbed = try_add_all(p_mask, perturbed)
        
        if len(perturbed) > best_size:
            best_result = perturbed
            best_size = len(perturbed)
            mask = p_mask
        
        # Local search if time permits
        if time.time() < DEADLINE - 10.0 and len(perturbed) >= best_size - 2:
            ls_set2, ls_mask2, ls_size2 = local_search(perturbed, p_mask, min(time.time() + 3.0, DEADLINE - 5.0))
            if ls_size2 > best_size:
                best_result = ls_set2
                best_size = ls_size2
                mask = ls_mask2
        
        if time.time() >= DEADLINE - 5.0:
            break
    
    return list(best_result)
