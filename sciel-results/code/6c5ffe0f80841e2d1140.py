import time
import random
import math

def construct():
    N = 256
    # Deadline: 90s limit, use 80s for safety
    deadline = time.time() + 80.0
    
    # 1. Precomputation
    # conflicts[x] is a list of masks. Each mask has bits set for the other 3 elements 
    # of a 4-AP that includes x.
    # If (current_mask & mask) == mask, adding x creates a 4-AP.
    conflicts = [[] for _ in range(N)]
    
    # triples[x] is a list of masks. Each mask has bits set for the other 2 elements 
    # of a 3-AP that includes x.
    # If (current_mask & mask) == mask, the 3-AP is complete.
    triples = [[] for _ in range(N)]
    
    # Generate all 4-APs
    # a, a+d, a+2d, a+3d
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # 4-AP: (a, b, c, e)
            # For a: others are b, c, e
            mask_a = (1 << b) | (1 << c) | (1 << e)
            conflicts[a].append(mask_a)
            
            # For b: others are a, c, e
            mask_b = (1 << a) | (1 << c) | (1 << e)
            conflicts[b].append(mask_b)
            
            # For c: others are a, b, e
            mask_c = (1 << a) | (1 << b) | (1 << e)
            conflicts[c].append(mask_c)
            
            # For e: others are a, b, c
            mask_e = (1 << a) | (1 << b) | (1 << c)
            conflicts[e].append(mask_e)
            
            # Also generate 3-APs from these and others
            # 3-APs: (a, b, c), (b, c, e), (a, c, e) is not a 3-AP (diffs d, 2d)
            # Wait, (a, a+2d, a+4d) is a 3-AP.
            # Let's generate 3-APs separately to be sure.
            
    # Generate all 3-APs
    # a, a+d, a+2d
    for a in range(N):
        max_d = (N - 1 - a) // 2
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            
            # 3-AP: (a, b, c)
            # For a: others are b, c
            mask_a = (1 << b) | (1 << c)
            triples[a].append(mask_a)
            
            # For b: others are a, c
            mask_b = (1 << a) | (1 << c)
            triples[b].append(mask_b)
            
            # For c: others are a, b
            mask_c = (1 << a) | (1 << b)
            triples[c].append(mask_c)

    # Helper functions
    def is_valid_add(mask, x):
        """Check if adding x to mask creates a 4-AP."""
        for cmask in conflicts[x]:
            if (mask & cmask) == cmask:
                return False
        return True

    def delta_3aps_add(mask, x):
        """Calculate increase in 3-AP count if x is added to mask."""
        delta = 0
        for tmask in triples[x]:
            if (mask & tmask) == tmask:
                delta += 1
        return delta

    def delta_3aps_remove(mask, x):
        """Calculate decrease in 3-AP count if x is removed from mask."""
        # Same logic: how many 3-APs are currently completed by x and other elements in mask?
        delta = 0
        for tmask in triples[x]:
            if (mask & tmask) == tmask:
                delta += 1
        return delta

    def count_3aps(mask):
        """Count total 3-APs. Used for initialization verification."""
        count = 0
        for x in range(N):
            if (mask >> x) & 1:
                count += delta_3aps_add(mask, x)
        # Each 3-AP is counted 3 times (once for each element)
        return count // 3

    # Greedy Initialization
    def greedy_init():
        mask = 0
        # Sort elements by how many 3-APs they can potentially form? 
        # Or just random order.
        elems = list(range(N))
        random.shuffle(elems)
        
        for x in elems:
            if is_valid_add(mask, x):
                # Add if it adds 3-APs, or with some probability to fill space
                d = delta_3aps_add(mask, x)
                if d > 0 or random.random() < 0.1:
                    mask |= (1 << x)
        return mask

    # Simulated Annealing
    best_mask = 0
    best_score = -1
    
    # Run multiple restarts
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
            
        current_mask = greedy_init()
        current_score = count_3aps(current_mask)
        
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
            
        T = 2.0
        T_min = 0.01
        cooling = 0.9995
        
        # Adjust steps based on time available for this restart
        # Estimate steps: we want to use ~90% of time_per_restart
        # Each step is fast (bit ops).
        # Let's just loop until time runs out for this restart or T is low.
        
        restart_deadline = time.time() + time_per_restart * 0.9
        
        while time.time() < restart_deadline:
            move_type = random.random()
            new_mask = current_mask
            new_score = current_score
            valid_move = False
            
            if move_type < 0.4:
                # Add
                x = random.randint(0, N - 1)
                if not ((current_mask >> x) & 1):
                    if is_valid_add(current_mask, x):
                        new_mask = current_mask | (1 << x)
                        new_score = current_score + delta_3aps_add(current_mask, x)
                        valid_move = True
            elif move_type < 0.7:
                # Remove
                x = random.randint(0, N - 1)
                if (current_mask >> x) & 1:
                    new_mask = current_mask & ~(1 << x)
                    new_score = current_score - delta_3aps_remove(current_mask, x)
                    valid_move = True
            else:
                # Swap
                x = random.randint(0, N - 1)
                if (current_mask >> x) & 1:
                    y = random.randint(0, N - 1)
                    if not ((current_mask >> y) & 1):
                        temp_mask = current_mask & ~(1 << x)
                        if is_valid_add(temp_mask, y):
                            d_rem = delta_3aps_remove(current_mask, x)
                            d_add = delta_3aps_add(temp_mask, y)
                            new_score = current_score - d_rem + d_add
                            new_mask = temp_mask | (1 << y)
                            valid_move = True

            if valid_move:
                if new_score >= current_score:
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                else:
                    delta = current_score - new_score
                    # Accept with probability exp(-delta / T)
                    if delta > 0:
                        # Use math.exp for precision, or approximation
                        if random.random() < math.exp(-delta / T):
                            current_mask = new_mask
                            current_score = new_score
                
                T *= cooling
                if T < T_min:
                    T = T_min

    # Convert best_mask back to sorted list (1-indexed)
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    return sorted(result)
