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
    bit_positions = [0] + [1 << (i - 1) for i in range(1, n + 1)]
    
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
    
    def greedy_fill(S_mask, S_list, order=None):
        """Fill with greedy in given order (or natural order if None)."""
        if order is None:
            order = list(range(1, n + 1))
        for x in order:
            if not (S_mask & bit_positions[x]):
                if can_add(x, S_mask):
                    S_mask |= bit_positions[x]
                    S_list.append(x)
        return S_mask, S_list
    
    best_mask = 0
    best_size = 0
    
    # Precompute all elements for faster iteration
    all_elements = list(range(1, n + 1))
    
    # Multiple greedy strategies with different orderings
    greedy_orders = [
        list(range(1, n + 1)),  # natural
        list(range(n, 0, -1)),  # reverse
    ]
    
    # Add some random orders
    for _ in range(3):
        shuffled = all_elements[:]
        random.shuffle(shuffled)
        greedy_orders.append(shuffled)
    
    # Initial greedy constructions
    for order in greedy_orders:
        S_mask = 0
        S_list = []
        S_mask, S_list = greedy_fill(S_mask, S_list, order)
        S_mask, S_list, _ = try_add_all(S_mask, S_list)
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
    
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
        
        # 3. Simulated annealing with swap moves and 2-opt
        # Temperature schedule: start high, cool down
        T_start = 8.0
        T_end = 0.02
        cooling_rate = 0.997
        max_sa_iterations = 80000
        T = T_start
        
        sa_iter = 0
        no_improve = 0
        max_no_improve = 500
        
        while sa_iter < max_sa_iterations and time.time() < deadline:
            if not S_list or not S_mask:
                break
            
            # Choose move type: 1-opt (70%), 2-opt (30%)
            move_type = random.random()
            
            if move_type < 0.7:
                # 1-opt: remove one, add one
                y_idx = random.randint(0, len(S_list) - 1)
                y = S_list[y_idx]
                
                # Pick a random element to add
                x = random.randint(1, n)
                if S_mask & bit_positions[x]:
                    sa_iter += 1
                    continue
                
                # Compute new mask after removing y and adding x
                S_mask_without_y = S_mask & ~bit_positions[y]
                
                if can_add(x, S_mask_without_y):
                    S_mask_new = S_mask_without_y | bit_positions[x]
                    
                    new_size = len(S_list)
                    
                    # Accept if better, or with probability exp(-delta/T) if worse
                    if new_size >= current_size:
                        S_mask = S_mask_new
                        S_list[y_idx] = x
                        current_size = new_size
                        no_improve = 0
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = S_mask
                    else:
                        # Accept worsening move with probability exp((new_size - current_size) / T)
                        delta = new_size - current_size  # negative
                        if random.random() < (2.718281828 ** (delta / T)):
                            S_mask = S_mask_new
                            S_list[y_idx] = x
                            current_size = new_size
                            no_improve += 1
                else:
                    no_improve += 1
            else:
                # 2-opt: remove two, add two
                if len(S_list) < 2:
                    sa_iter += 1
                    continue
                    
                y1_idx = random.randint(0, len(S_list) - 1)
                y2_idx = random.randint(0, len(S_list) - 1)
                if y1_idx == y2_idx:
                    sa_iter += 1
                    continue
                    
                y1 = S_list[y1_idx]
                y2 = S_list[y2_idx]
                
                # Pick two random elements to add
                x1 = random.randint(1, n)
                x2 = random.randint(1, n)
                if (S_mask & bit_positions[x1]) or (S_mask & bit_positions[x2]):
                    sa_iter += 1
                    continue
                if x1 == x2:
                    sa_iter += 1
                    continue
                
                # Compute new mask after removing y1, y2 and adding x1, x2
                S_mask_without = S_mask & ~bit_positions[y1] & ~bit_positions[y2]
                
                if can_add(x1, S_mask_without) and can_add(x2, S_mask_without | bit_positions[x1]):
                    S_mask_new = S_mask_without | bit_positions[x1] | bit_positions[x2]
                    
                    new_size = len(S_list)
                    
                    # Accept if better, or with probability exp(-delta/T) if worse
                    if new_size >= current_size:
                        S_mask = S_mask_new
                        S_list[y1_idx] = x1
                        S_list[y2_idx] = x2
                        current_size = new_size
                        no_improve = 0
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = S_mask
                    else:
                        # Accept worsening move with probability exp((new_size - current_size) / T)
                        delta = new_size - current_size  # negative
                        if random.random() < (2.718281828 ** (delta / T)):
                            S_mask = S_mask_new
                            S_list[y1_idx] = x1
                            S_list[y2_idx] = x2
                            current_size = new_size
                            no_improve += 1
                else:
                    no_improve += 1
            
            # Every 500 iterations, do a full add pass to recover
            if sa_iter % 500 == 0:
                S_mask, S_list, _ = try_add_all(S_mask, S_list)
                current_size = len(S_list)
                if current_size > best_size:
                    best_size = current_size
                    best_mask = S_mask
                    no_improve = 0
            
            # Cool down
            T *= cooling_rate
            if T < T_end:
                T = T_end
            
            sa_iter += 1
            
            # If stuck for too long, restart SA with current best
            if no_improve > max_no_improve:
                # Reset to best and restart SA
                S_list = []
                for i in range(n):
                    if best_mask & (1 << i):
                        S_list.append(i + 1)
                S_mask = best_mask
                current_size = best_size
                T = T_start
                no_improve = 0
        
        # Final add pass after SA
        S_mask, S_list, _ = try_add_all(S_mask, S_list)
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
    
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
