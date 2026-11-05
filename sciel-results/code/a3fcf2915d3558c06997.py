"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 512

def _precompute_conflicts():
    """
    Precompute conflict masks for each element.
    conflicts[x] is a list of masks. Each mask is a 512-bit integer representing
    a set of 3 elements {a, b, c} such that {x, a, b, c} forms a 4-term AP.
    If x is in S, then for all masks in conflicts[x], (S & mask) != mask must hold.
    """
    conflicts = [[] for _ in range(N + 1)]
    
    # Generate all 4-term APs: p, p+d, p+2d, p+3d
    for p in range(1, N + 1):
        max_d = (N - p) // 3
        for d in range(1, max_d + 1):
            t1 = p
            t2 = p + d
            t3 = p + 2 * d
            t4 = p + 3 * d
            
            # For each element, the other 3 form a forbidden mask
            # If t1 is in S, {t2, t3, t4} cannot all be in S.
            mask1 = (1 << t2) | (1 << t3) | (1 << t4)
            conflicts[t1].append(mask1)
            
            mask2 = (1 << t1) | (1 << t3) | (1 << t4)
            conflicts[t2].append(mask2)
            
            mask3 = (1 << t1) | (1 << t2) | (1 << t4)
            conflicts[t3].append(mask3)
            
            mask4 = (1 << t1) | (1 << t2) | (1 << t3)
            conflicts[t4].append(mask4)
            
    return conflicts

def construct():
    random.seed(42)
    conflicts = _precompute_conflicts()
    
    # Precompute degrees for heuristic initialization
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        degrees[i] = len(conflicts[i])

    def can_add(x, S_mask):
        # Inline check for speed
        for mask in conflicts[x]:
            if (S_mask & mask) == mask:
                return False
        return True

    # 1. Greedy Initialization
    best_S = 0
    best_size = 0
    best_list = []
    
    # Strategy 1: Random shuffles
    for _ in range(20):
        current_S = 0
        current_list = []
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        for x in order:
            if can_add(x, current_S):
                current_S |= (1 << x)
                current_list.append(x)
        
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_S = current_S
            best_list = current_list[:]

    # Strategy 2: Greedy by degree (ascending - pick elements with fewest conflicts first)
    order_deg = sorted(range(1, N + 1), key=lambda x: degrees[x])
    current_S = 0
    current_list = []
    for x in order_deg:
        if can_add(x, current_S):
            current_S |= (1 << x)
            current_list.append(x)
    if len(current_list) > best_size:
        best_size = len(current_list)
        best_S = current_S
        best_list = current_list[:]

    # Strategy 3: Greedy by degree (descending - pick elements with most conflicts first? usually bad, but let's try)
    order_deg_desc = sorted(range(1, N + 1), key=lambda x: -degrees[x])
    current_S = 0
    current_list = []
    for x in order_deg_desc:
        if can_add(x, current_S):
            current_S |= (1 << x)
            current_list.append(x)
    if len(current_list) > best_size:
        best_size = len(current_list)
        best_S = current_S
        best_list = current_list[:]

    S = best_S
    S_list = best_list[:]
    current_size = best_size
    best_size_found = best_size
    best_S_found = S
    best_list_found = S_list[:]

    # 2. Simulated Annealing
    time_limit = 55.0
    start_time = time.time()
    
    T = 50.0
    T_min = 0.01
    cooling_rate = 0.99995
    
    all_elements = list(range(1, N + 1))
    
    # Local variables for speed
    conf = conflicts
    can_add_func = can_add

    while time.time() - start_time < time_limit:
        T *= cooling_rate
        if T < T_min:
            T = T_min

        move_type = random.random()
        
        if move_type < 0.3:
            # Try to Add
            # Pick random element not in S
            # Efficient way: pick random index, if in S, retry or just skip? 
            # Better: maintain a list of not-in-S elements? 
            # For N=512, random.choice(all_elements) and check is fine.
            x = random.choice(all_elements)
            if not (S & (1 << x)):
                if can_add_func(x, S):
                    S |= (1 << x)
                    S_list.append(x)
                    current_size += 1
                    if current_size > best_size_found:
                        best_size_found = current_size
                        best_S_found = S
                        best_list_found = S_list[:]
        
        elif move_type < 0.6:
            # Try to Swap (1-out, 1-in)
            if S_list:
                x = random.choice(S_list)
                y = random.choice(all_elements)
                if x != y and not (S & (1 << y)):
                    S_wo_x = S & ~(1 << x)
                    if can_add_func(y, S_wo_x):
                        S = S_wo_x | (1 << y)
                        idx = S_list.index(x)
                        S_list[idx] = y
                        # Size unchanged, but structure changed. 
                        # We don't update best here because size is same, 
                        # but this helps escape local minima for future adds.
        
        elif move_type < 0.8:
            # Try to Remove
            if S_list:
                x = random.choice(S_list)
                # Accept removal with probability exp(-1/T)
                if random.random() < 2.71828 ** (-1.0 / T):
                    S &= ~(1 << x)
                    idx = S_list.index(x)
                    S_list.pop(idx)
                    current_size -= 1
                    
        else:
            # Try 2-Swap (2-out, 2-in)
            if len(S_list) >= 2:
                # Pick two distinct elements to remove
                idx1 = random.randint(0, len(S_list) - 1)
                idx2 = random.randint(0, len(S_list) - 1)
                if idx1 != idx2:
                    x1 = S_list[idx1]
                    x2 = S_list[idx2]
                    
                    # Pick two distinct elements to add
                    y1 = random.choice(all_elements)
                    y2 = random.choice(all_elements)
                    
                    # Ensure y1, y2 are not in S and distinct from each other
                    if (not (S & (1 << y1))) and (not (S & (1 << y2))) and y1 != y2:
                        # Also ensure y1, y2 are not x1, x2 (they can't be, since x1,x2 in S and y1,y2 not in S)
                        S_wo = S & ~(1 << x1) & ~(1 << x2)
                        
                        if can_add_func(y1, S_wo):
                            S_tmp = S_wo | (1 << y1)
                            if can_add_func(y2, S_tmp):
                                S = S_tmp | (1 << y2)
                                # Update S_list
                                # Remove x1, x2
                                # Note: indices might have shifted if we just pop. 
                                # Safer to rebuild or swap-remove.
                                # Since this is rare, we can rebuild S_list from S? No, too slow.
                                # Let's do careful list manipulation.
                                # Find indices in current S_list
                                # x1 and x2 are in S_list.
                                # We need to replace them with y1 and y2.
                                # This is complex to do in-place with indices.
                                # Alternative: Just mark as accepted and fix list later? 
                                # Or, simpler: Just accept the move in S, and fix S_list.
                                # Since S_list is only used for sampling, we can just:
                                # 1. Find index of x1, replace with y1.
                                # 2. Find index of x2, replace with y2.
                                # But what if y1 == y2? We checked distinct.
                                # What if y1 == x2? Impossible, x2 in S, y1 not in S.
                                
                                try:
                                    i1 = S_list.index(x1)
                                    S_list[i1] = y1
                                    i2 = S_list.index(x2)
                                    S_list[i2] = y2
                                except ValueError:
                                    pass # Should not happen

        # Periodic Greedy Fill (repair)
        if random.random() < 0.05:
            # Try to add any valid element
            # This helps recover from bad removals
            for x in all_elements:
                if not (S & (1 << x)):
                    if can_add_func(x, S):
                        S |= (1 << x)
                        S_list.append(x)
                        current_size += 1
                        if current_size > best_size_found:
                            best_size_found = current_size
                            best_S_found = S
                            best_list_found = S_list[:]

    return sorted(best_list_found)

# EVOLVE-BLOCK-END
