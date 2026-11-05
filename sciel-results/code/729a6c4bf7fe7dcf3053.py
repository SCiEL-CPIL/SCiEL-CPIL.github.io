import time
import random

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
    
    def try_add_all(S_mask, S_list):
        """Try to add all possible elements in natural order. Returns (new_mask, new_list)."""
        added = 0
        for x in range(1, n + 1):
            if not (S_mask & bit_positions[x]):
                if can_add(x, S_mask):
                    S_mask |= bit_positions[x]
                    S_list.append(x)
                    added += 1
        return S_mask, S_list

    best_mask = 0
    best_size = 0
    all_elements = list(range(1, n + 1))
    
    # 1. Initial Greedy Constructions to find a good starting point
    # Try natural, reverse, and a few random orders
    greedy_orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
    ]
    for _ in range(5):
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
        
        # Polish
        S_mask, S_list = try_add_all(S_mask, S_list)
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
    
    # If we haven't found anything (shouldn't happen), return trivial
    if best_size == 0:
        return [1]

    # 2. Simulated Annealing to improve the best solution
    # We run SA from the current best state
    S_list = []
    for i in range(n):
        if best_mask & (1 << i):
            S_list.append(i + 1)
    S_mask = best_mask
    current_size = best_size
    
    # SA Parameters
    T_start = 10.0
    T_end = 0.05
    cooling_rate = 0.9995
    max_sa_iterations = 200000  # Cap for safety, but time is the real limit
    sa_iter = 0
    no_improve = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        if not S_list:
            break
            
        # Choose move type: 1-opt (80%), 2-opt (20%)
        # 1-opt is faster and often sufficient for small improvements
        # 2-opt helps escape deeper local minima
        
        if random.random() < 0.8:
            # 1-opt: Remove y, Add x
            y_idx = random.randint(0, len(S_list) - 1)
            y = S_list[y_idx]
            y_bit = bit_positions[y]
            
            # Pick random x not in S
            # Optimization: Instead of checking all, just pick random and retry a few times
            x_added = False
            for _ in range(10):
                x = random.randint(1, n)
                if S_mask & bit_positions[x]:
                    continue
                    
                S_mask_without_y = S_mask & ~y_bit
                if can_add(x, S_mask_without_y):
                    S_mask_new = S_mask_without_y | bit_positions[x]
                    S_list[y_idx] = x
                    S_mask = S_mask_new
                    # Size stays same, so it's a neutral move for cardinality
                    # But it changes the structure. We accept neutral moves to explore.
                    # However, for maximizing cardinality, we really want to find states
                    # where we can add MORE. 
                    # So we accept neutral moves with some probability, or always?
                    # Standard SA accepts neutral. 
                    # But to be aggressive on size, we might prefer moves that
                    # allow future additions. 
                    # Let's accept neutral moves to escape local optima.
                    x_added = True
                    # We don't update no_improve for neutral moves immediately, 
                    # because we hope this leads to a size increase later.
                    # But if we do many neutral moves without size increase, we should cool down.
                    break
            
            if not x_added:
                # No valid swap found
                no_improve += 1
                # Cool down
                T = T_start * (cooling_rate ** sa_iter)
                if T < T_end: T = T_end
                sa_iter += 1
                if no_improve > max_no_improve:
                    # Restart from best
                    S_list = []
                    for i in range(n):
                        if best_mask & (1 << i):
                            S_list.append(i + 1)
                    S_mask = best_mask
                    current_size = best_size
                    no_improve = 0
                    sa_iter = 0
                continue
        else:
            # 2-opt: Remove y1, y2; Add x1, x2
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
            y1_bit = bit_positions[y1]
            y2_bit = bit_positions[y2]
            
            x1_added = False
            for _ in range(10):
                x1 = random.randint(1, n)
                x2 = random.randint(1, n)
                
                if x1 == x2: continue
                if S_mask & bit_positions[x1]: continue
                if S_mask & bit_positions[x2]: continue
                
                S_mask_without = S_mask & ~y1_bit & ~y2_bit
                
                if can_add(x1, S_mask_without) and can_add(x2, S_mask_without | bit_positions[x1]):
                    S_mask_new = S_mask_without | bit_positions[x1] | bit_positions[x2]
                    S_list[y1_idx] = x1
                    S_list[y2_idx] = x2
                    S_mask = S_mask_new
                    x1_added = True
                    break
            
            if not x1_added:
                no_improve += 1
                T = T_start * (cooling_rate ** sa_iter)
                if T < T_end: T = T_end
                sa_iter += 1
                if no_improve > max_no_improve:
                    S_list = []
                    for i in range(n):
                        if best_mask & (1 << i):
                            S_list.append(i + 1)
                    S_mask = best_mask
                    current_size = best_size
                    no_improve = 0
                    sa_iter = 0
                continue
        
        # After a move (neutral), try to add more elements (greedy polish)
        # This is the key to increasing cardinality.
        # If the swap opened a hole, we might be able to add 1 or more elements.
        
        # Periodic full greedy pass is expensive but necessary to find global adds.
        # We do a light pass every few iterations, and a full pass when stuck or periodically.
        
        if sa_iter % 10 == 0:
            # Light pass: try to add a few random elements
            for _ in range(5):
                x = random.randint(1, n)
                if not (S_mask & bit_positions[x]):
                    if can_add(x, S_mask):
                        S_mask |= bit_positions[x]
                        S_list.append(x)
                        current_size += 1
                        no_improve = 0
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = S_mask
        
        # Every 500 iterations, do a full greedy fill
        if sa_iter % 500 == 0:
            S_mask, S_list = try_add_all(S_mask, S_list)
            current_size = len(S_list)
            if current_size > best_size:
                best_size = current_size
                best_mask = S_mask
                no_improve = 0
        
        # Update SA state
        sa_iter += 1
        
    # Final polish
    final_list = []
    for i in range(n):
        if best_mask & (1 << i):
            final_list.append(i + 1)
    
    # One last try to add anything
    temp_mask = best_mask
    temp_list = final_list[:]
    temp_mask, temp_list = try_add_all(temp_mask, temp_list)
    if len(temp_list) > best_size:
        best_size = len(temp_list)
        best_mask = temp_mask
        final_list = temp_list

    if not final_list:
        return [1]
        
    return final_list
