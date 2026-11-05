import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin for 60s limit
    
    n = 512
    
    # Precompute all 4-APs
    # For each element x, conflict_masks[x] is a list of masks.
    # If x is in S, and (mask & S_mask) == mask, then a 4-AP exists.
    # The mask contains the other 3 elements.
    conflict_masks = [[] for _ in range(n + 1)]
    
    # Generate all 4-APs (a, a+d, a+2d, a+3d)
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            term4 = a + 3 * d
            
            # For each element in the AP, the other 3 form the conflict mask
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_masks[a].append(mask_a)
            
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_masks[b].append(mask_b)
            
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (term4 - 1))
            conflict_masks[c].append(mask_c)
            
            mask_d = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflict_masks[term4].append(mask_d)
            
    # Convert to tuples for faster iteration
    conflict_masks = [tuple(lst) for lst in conflict_masks]
    
    bit_positions = [0] + [1 << (i - 1) for i in range(1, n + 1)]
    
    def can_add(x, S_mask):
        """Check if x can be added to S without forming a 4-AP."""
        for mask in conflict_masks[x]:
            if (mask & S_mask) == mask:
                return False
        return True

    def greedy_fill(S_mask, S_list, order):
        """Try to add all possible elements in the given order."""
        added_any = False
        for x in order:
            if not (S_mask & bit_positions[x]):
                if can_add(x, S_mask):
                    S_mask |= bit_positions[x]
                    S_list.append(x)
                    added_any = True
        return S_mask, S_list, added_any

    def salem_spencer_seed(start, length, S_mask):
        """
        Generate a Salem-Spencer-like seed for a block.
        Uses base-3 digits to create a 3-AP-free (hence 4-AP-free) set.
        """
        candidates = []
        for x in range(start, min(start + length, n + 1)):
            # Convert to base 3 and check if all digits are in {0, 1}
            # This creates a set with no 3-AP, hence no 4-AP
            temp = x - start
            valid = True
            while temp > 0:
                if temp % 3 == 2:
                    valid = False
                    break
                temp //= 3
            if valid:
                candidates.append(x)
        
        # Shuffle for diversity
        random.shuffle(candidates)
        
        local_seed = []
        for x in candidates:
            if not (S_mask & bit_positions[x]):
                if can_add(x, S_mask):
                    S_mask |= bit_positions[x]
                    local_seed.append(x)
        
        return local_seed, S_mask

    best_mask = 0
    best_size = 0
    all_elements = list(range(1, n + 1))
    
    # --- Strategy: Multi-phase construction ---
    # Phase 1: Salem-Spencer seed
    # Phase 2: Greedy fill
    # Phase 3: Local search with swaps
    
    # Run multiple restarts to find a good solution
    while time.time() < deadline:
        S_mask = 0
        S_list = []
        
        # 1. Salem-Spencer Seed Phase
        # Generate a base-3 pattern seed
        ss_seed, S_mask = salem_spencer_seed(1, n, S_mask)
        S_list.extend(ss_seed)
        
        # 2. Greedy Fill Phase
        # Try to add more elements in a random order
        random_order = all_elements[:]
        random.shuffle(random_order)
        S_mask, S_list, _ = greedy_fill(S_mask, S_list, random_order)
        
        current_size = len(S_list)
        
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
            
        # 3. Local Search Phase
        # Use simulated annealing with swaps to improve
        
        no_improvement_count = 0
        max_no_improvement = 50000
        step = 0
        
        while time.time() < deadline and no_improvement_count < max_no_improvement:
            step += 1
            
            # Try to add an element
            added = False
            for _ in range(3):
                x = random.randint(1, n)
                if not (S_mask & bit_positions[x]):
                    if can_add(x, S_mask):
                        S_mask |= bit_positions[x]
                        S_list.append(x)
                        current_size += 1
                        added = True
                        no_improvement_count = 0
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = S_mask
                        break
            
            if added:
                continue
                
            # If no element could be added, try a swap
            # Pick random y in S to remove, then try to add x
            if S_list:
                y_idx = random.randint(0, len(S_list) - 1)
                y = S_list[y_idx]
                
                # Try multiple x candidates
                swap_made = False
                for _ in range(15):
                    x = random.randint(1, n)
                    if S_mask & bit_positions[x]:
                        continue
                    
                    # Check if x can be added to S \ {y}
                    S_mask_without_y = S_mask & ~bit_positions[y]
                    if can_add(x, S_mask_without_y):
                        # Perform swap
                        S_mask = S_mask_without_y | bit_positions[x]
                        S_list[y_idx] = x
                        swap_made = True
                        no_improvement_count = 0
                        break
                
                if not swap_made:
                    no_improvement_count += 1
                
                # Periodic fill to recover from any potential "holes"
                if no_improvement_count % 200 == 0:
                    S_mask, S_list, _ = greedy_fill(S_mask, S_list, all_elements)
                    new_size = len(S_list)
                    if new_size > current_size:
                        current_size = new_size
                        no_improvement_count = 0
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = S_mask

        # Final fill for this restart
        S_mask, S_list, _ = greedy_fill(S_mask, S_list, all_elements)
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask

    # Final polish: try to add any remaining elements to the best solution
    # Reconstruct list from best_mask
    final_list = []
    for i in range(1, n + 1):
        if best_mask & bit_positions[i]:
            final_list.append(i)
    
    if not final_list:
        return [1, 2]
        
    # One last attempt to fill
    final_mask = best_mask
    for x in range(1, n + 1):
        if not (final_mask & bit_positions[x]):
            if can_add(x, final_mask):
                final_mask |= bit_positions[x]
                final_list.append(x)
    
    if not final_list:
        return [1, 2]
        
    return final_list
