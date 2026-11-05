import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit
    
    n = 512
    
    # Precompute all 4-APs
    # For each x, store a list of masks representing the other 3 elements of APs containing x
    conflict_triples = [[] for _ in range(n + 1)]
    
    # Generate all 4-APs (a, a+d, a+2d, a+3d)
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            term4 = a + 3 * d
            
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_triples[a].append(mask_a)
            
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_triples[b].append(mask_b)
            
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (term4 - 1))
            conflict_triples[c].append(mask_c)
            
            mask_d = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflict_triples[term4].append(mask_d)
    
    # Convert to tuples for faster iteration
    conflict_triples = [tuple(lst) for lst in conflict_triples]
    
    # Precompute bit positions as a list for faster indexing (1-indexed)
    bit_positions = [0] * (n + 1)
    for i in range(1, n + 1):
        bit_positions[i] = 1 << (i - 1)
    
    def can_add(x, S_mask):
        """Check if x can be added to S without forming a 4-AP."""
        for mask in conflict_triples[x]:
            if (mask & S_mask) == mask:
                return False
        return True
    
    def try_add_all(S_mask, S_list, order=None):
        """Try to add all possible elements. Returns (new_mask, new_list)."""
        if order is None:
            order = list(range(1, n + 1))
        added = 0
        for x in order:
            if not (S_mask & bit_positions[x]):
                if can_add(x, S_mask):
                    S_mask |= bit_positions[x]
                    S_list.append(x)
                    added += 1
        return S_mask, S_list
    
    def greedy_fill(S_mask, S_list, order=None):
        """Greedy fill from a given state. Returns (new_mask, new_list)."""
        if order is None:
            order = list(range(1, n + 1))
        for x in order:
            if not (S_mask & bit_positions[x]):
                if can_add(x, S_mask):
                    S_mask |= bit_positions[x]
                    S_list.append(x)
        return S_mask, S_list
    
    def get_list_from_mask(mask):
        """Convert mask to list."""
        result = []
        for i in range(n):
            if mask & (1 << i):
                result.append(i + 1)
        return result
    
    best_mask = 0
    best_size = 0
    all_elements = list(range(1, n + 1))
    
    # 1. Initial Greedy Constructions to find a good starting point
    # Try multiple deterministic and random orderings
    greedy_orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
    ]
    
    # Add AP-count sorted ordering: elements with fewer 4-AP conflicts first
    ap_counts = [len(conflict_triples[i]) for i in range(1, n + 1)]
    ap_sorted = sorted(range(1, n + 1), key=lambda x: ap_counts[x])
    greedy_orders.append(ap_sorted)
    
    # Add a few random shuffles
    for _ in range(8):
        shuffled = all_elements[:]
        random.shuffle(shuffled)
        greedy_orders.append(shuffled)
        
    for order in greedy_orders:
        S_mask = 0
        S_list = []
        for x in order:
            if can_add(x, S_mask):
                S_mask |= bit_positions[x]
                S_list.append(x)
        
        # Polish with natural order
        S_mask, S_list = try_add_all(S_mask, S_list, list(range(1, n + 1)))
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
    
    # If we haven't found anything (shouldn't happen), return trivial
    if best_size == 0:
        return [1]

    # 2. Local Search: 1-exchange and 2-exchange moves
    # We run from the current best state
    
    def local_search(start_mask, start_list, max_iters=50000):
        """Perform local search with 1-exchange and 2-exchange moves."""
        S_list = start_list[:]
        S_mask = start_mask
        current_size = len(S_list)
        best_local_mask = S_mask
        best_local_size = current_size
        best_local_list = S_list[:]
        
        iters = 0
        no_improve = 0
        max_no_improve = 2000
        
        while iters < max_iters and time.time() < deadline:
            improved = False
            
            # 1-exchange: Remove y, Add x
            # Try to find a swap that increases size by 1
            # Or a neutral swap that enables future additions
            
            # First, try to find an element we can add directly
            added_any = False
            for x in range(1, n + 1):
                if not (S_mask & bit_positions[x]):
                    if can_add(x, S_mask):
                        S_mask |= bit_positions[x]
                        S_list.append(x)
                        current_size += 1
                        added_any = True
                        improved = True
                        break
            
            if added_any:
                no_improve = 0
                if current_size > best_local_size:
                    best_local_size = current_size
                    best_local_mask = S_mask
                    best_local_list = S_list[:]
                iters += 1
                continue
            
            # Try 1-exchange moves
            found_swap = False
            for _ in range(20):
                y_idx = random.randint(0, len(S_list) - 1)
                y = S_list[y_idx]
                y_bit = bit_positions[y]
                
                # Try a few random x
                for _ in range(5):
                    x = random.randint(1, n)
                    if S_mask & bit_positions[x]:
                        continue
                    
                    S_mask_without_y = S_mask & ~y_bit
                    if can_add(x, S_mask_without_y):
                        # Check if this enables adding more
                        S_mask_new = S_mask_without_y | bit_positions[x]
                        
                        # Try to add more after this swap
                        extra_added = 0
                        temp_mask = S_mask_new
                        for z in range(1, n + 1):
                            if not (temp_mask & bit_positions[z]):
                                if can_add(z, temp_mask):
                                    temp_mask |= bit_positions[z]
                                    extra_added += 1
                                    if extra_added >= 1:
                                        break
                        
                        if extra_added > 0:
                            # Accept this swap
                            S_mask = temp_mask
                            S_list[y_idx] = x
                            # Add the extra elements
                            S_list = get_list_from_mask(S_mask)
                            current_size = len(S_list)
                            found_swap = True
                            improved = True
                            break
                    
                    if found_swap:
                        break
                
                if found_swap:
                    break
            
            if found_swap:
                no_improve = 0
                if current_size > best_local_size:
                    best_local_size = current_size
                    best_local_mask = S_mask
                    best_local_list = S_list[:]
                iters += 1
                continue
            
            # Try 2-exchange moves (remove 2, add 2)
            if len(S_list) >= 2:
                found_2swap = False
                for _ in range(10):
                    y1_idx = random.randint(0, len(S_list) - 1)
                    y2_idx = random.randint(0, len(S_list) - 1)
                    if y1_idx == y2_idx:
                        continue
                    
                    y1 = S_list[y1_idx]
                    y2 = S_list[y2_idx]
                    y1_bit = bit_positions[y1]
                    y2_bit = bit_positions[y2]
                    
                    for _ in range(3):
                        x1 = random.randint(1, n)
                        x2 = random.randint(1, n)
                        
                        if x1 == x2:
                            continue
                        if S_mask & bit_positions[x1]:
                            continue
                        if S_mask & bit_positions[x2]:
                            continue
                        
                        S_mask_without = S_mask & ~y1_bit & ~y2_bit
                        
                        if can_add(x1, S_mask_without) and can_add(x2, S_mask_without | bit_positions[x1]):
                            S_mask_new = S_mask_without | bit_positions[x1] | bit_positions[x2]
                            
                            # Try to add more
                            extra_added = 0
                            temp_mask = S_mask_new
                            for z in range(1, n + 1):
                                if not (temp_mask & bit_positions[z]):
                                    if can_add(z, temp_mask):
                                        temp_mask |= bit_positions[z]
                                        extra_added += 1
                                        if extra_added >= 1:
                                            break
                            
                            if extra_added > 0:
                                S_mask = temp_mask
                                S_list = get_list_from_mask(S_mask)
                                current_size = len(S_list)
                                found_2swap = True
                                improved = True
                                break
                        
                        if found_2swap:
                            break
                    
                    if found_2swap:
                        break
                
                if found_2swap:
                    no_improve = 0
                    if current_size > best_local_size:
                        best_local_size = current_size
                        best_local_mask = S_mask
                        best_local_list = S_list[:]
                    iters += 1
                    continue
            
            # No improvement found
            no_improve += 1
            iters += 1
            
            if no_improve > max_no_improve:
                break
        
        return best_local_mask, best_local_list, best_local_size
    
    # Run local search multiple times with different random seeds
    for ls_round in range(5):
        if time.time() >= deadline:
            break
        
        # Start from best
        S_list = get_list_from_mask(best_mask)
        
        # Add some randomness by perturbing the initial state
        if ls_round > 0 and len(S_list) > 10:
            # Remove a few random elements and re-add greedily
            perturb_mask = best_mask
            perturb_list = S_list[:]
            for _ in range(min(5, len(perturb_list) // 2)):
                idx = random.randint(0, len(perturb_list) - 1)
                y = perturb_list[idx]
                perturb_mask &= ~bit_positions[y]
                perturb_list.pop(idx)
            perturb_mask, perturb_list = greedy_fill(perturb_mask, perturb_list)
            S_list = perturb_list
        
        ls_mask, ls_list, ls_size = local_search(best_mask if ls_round == 0 else perturb_mask, S_list)
        
        if ls_size > best_size:
            best_size = ls_size
            best_mask = ls_mask
    
    # Final polish
    final_list = get_list_from_mask(best_mask)
    
    # One last try to add anything
    temp_mask = best_mask
    temp_list = final_list[:]
    temp_mask, temp_list = try_add_all(temp_mask, temp_list, list(range(1, n + 1)))
    if len(temp_list) > best_size:
        best_size = len(temp_list)
        best_mask = temp_mask
        final_list = temp_list

    if not final_list:
        return [1]
        
    return final_list

