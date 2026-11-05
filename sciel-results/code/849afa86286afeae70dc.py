import time
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 58  # 2s safety margin for 60s limit
    
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
    
    # Precompute bit positions for quick lookup
    bit_positions = {i: 1 << (i - 1) for i in range(1, n + 1)}
    
    def can_add(x, S_mask):
        """Check if x can be added to S without forming a 4-AP."""
        for mask in conflict_triples[x]:
            if (mask & S_mask) == mask:
                return False
        return True
    
    def try_add_all(S_mask, S_list):
        """Try to add all possible elements. Returns (new_mask, new_list, added_count)."""
        added = 0
        for x in range(1, n + 1):
            if not (S_mask & bit_positions[x]):
                if can_add(x, S_mask):
                    S_mask |= bit_positions[x]
                    S_list.append(x)
                    added += 1
        return S_mask, S_list, added
    
    best_mask = 0
    best_size = 0
    
    # Precompute all elements for faster iteration
    all_elements = list(range(1, n + 1))
    
    while time.time() < deadline:
        # 1. Greedy Construction with random shuffle
        S_mask = 0
        S_list = []
        candidates = all_elements[:]
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S_mask):
                S_mask |= bit_positions[x]
                S_list.append(x)
        
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
        
        # 2. Exhaustive add pass
        S_mask, S_list, _ = try_add_all(S_mask, S_list)
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
        
        # 3. Local search with swaps - improved version
        # Try multiple swap candidates per iteration for better exploration
        max_swap_rounds = 5000
        no_improve_count = 0
        max_no_improve = 200
        
        for round_num in range(max_swap_rounds):
            if time.time() >= deadline:
                break
            
            if not S_list:
                break
            
            # Check time more frequently
            if round_num % 100 == 0 and time.time() >= deadline:
                break
            
            # Try up to 3 different swap candidates per round
            found_swap = False
            for attempt in range(3):
                x = random.randint(1, n)
                if S_mask & bit_positions[x]:
                    continue
                
                # Pick y in S uniformly
                y = random.choice(S_list)
                
                # Try swap: remove y, add x
                S_mask_without_y = S_mask & ~bit_positions[y]
                
                if can_add(x, S_mask_without_y):
                    # Accept swap
                    S_mask = S_mask_without_y | bit_positions[x]
                    # Update S_list: replace y with x
                    idx = S_list.index(y)
                    S_list[idx] = x
                    
                    # After swap, try to add more elements
                    S_mask, S_list, added = try_add_all(S_mask, S_list)
                    current_size = len(S_list)
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = S_mask
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    found_swap = True
                    break
            
            # If no swap was found, count as no improvement
            if not found_swap:
                no_improve_count += 1
            
            # Early exit if stuck
            if no_improve_count > max_no_improve:
                break
    
    # Final polish: try to add any remaining elements
    final_list = []
    for i in range(n):
        if best_mask & (1 << i):
            final_list.append(i + 1)
    
    if final_list:
        # Try one more add pass on the best solution
        temp_mask = best_mask
        temp_list = final_list[:]
        temp_mask, temp_list, _ = try_add_all(temp_mask, temp_list)
        if len(temp_list) > best_size:
            best_size = len(temp_list)
            best_mask = temp_mask
            final_list = temp_list
    
    if not final_list:
        return [1, 2]
        
    return final_list
