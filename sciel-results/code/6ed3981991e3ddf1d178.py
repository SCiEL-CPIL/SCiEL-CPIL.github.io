import time
import random

N = 256
DEADLINE = 85.0  # 90s limit, keep 5s safety margin

def _precompute():
    """Precompute 3-APs and 4-APs with bitmask representations for fast operations."""
    # 3-APs: (a, a+d, a+2d)
    aps3 = []
    aps3_other_masks = []  # For each 3-AP, store bitmask of the other two elements for each position
    
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            idx = len(aps3)
            aps3.append((a, b, c))
            # For element a: other two are b, c
            # For element b: other two are a, c  
            # For element c: other two are a, b
            aps3_other_masks.append([
                (1 << b) | (1 << c),  # if x=a, check b and c
                (1 << a) | (1 << c),  # if x=b, check a and c
                (1 << a) | (1 << b),  # if x=c, check a and b
            ])
    
    # 4-APs: (a, a+d, a+2d, a+3d)
    aps4 = []
    aps4_other_masks = []  # For each 4-AP, store bitmask of other three for each position
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            idx = len(aps4)
            aps4.append((a, b, c, e))
            # For element a: other three are b, c, e
            # For element b: other three are a, c, e
            # For element c: other three are a, b, e
            # For element e: other three are a, b, c
            aps4_other_masks.append([
                (1 << b) | (1 << c) | (1 << e),  # if x=a
                (1 << a) | (1 << c) | (1 << e),  # if x=b
                (1 << a) | (1 << b) | (1 << e),  # if x=c
                (1 << a) | (1 << b) | (1 << c),  # if x=e
            ])
    
    # Map element to list of (ap_index, position_in_ap)
    elem_to_aps3 = [[] for _ in range(N + 1)]
    for i, (a, b, c) in enumerate(aps3):
        elem_to_aps3[a].append((i, 0))
        elem_to_aps3[b].append((i, 1))
        elem_to_aps3[c].append((i, 2))
    
    elem_to_aps4 = [[] for _ in range(N + 1)]
    for i, (a, b, c, e) in enumerate(aps4):
        elem_to_aps4[a].append((i, 0))
        elem_to_aps4[b].append((i, 1))
        elem_to_aps4[c].append((i, 2))
        elem_to_aps4[e].append((i, 3))
    
    return aps3, aps4, aps3_other_masks, aps4_other_masks, elem_to_aps3, elem_to_aps4

def _initial_greedy_set():
    """Greedy construction for no 4-AP using bitmask."""
    selected_mask = 0
    for x in range(1, N + 1):
        x_bit = 1 << x
        valid = True
        # x could be the 1st, 2nd, 3rd, or 4th term
        # 1st: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            if ((1 << (x + d)) & selected_mask) and ((1 << (x + 2*d)) & selected_mask) and ((1 << (x + 3*d)) & selected_mask):
                valid = False
                break
        if not valid: continue
        # 2nd: x-d, x, x+d, x+2d
        for d in range(1, min(x - 1, (N - x) // 2) + 1):
            if ((1 << (x - d)) & selected_mask) and ((1 << (x + d)) & selected_mask) and ((1 << (x + 2*d)) & selected_mask):
                valid = False
                break
        if not valid: continue
        # 3rd: x-2d, x-d, x, x+d
        for d in range(1, min(x // 2, N - x) + 1):
            if ((1 << (x - 2*d)) & selected_mask) and ((1 << (x - d)) & selected_mask) and ((1 << (x + d)) & selected_mask):
                valid = False
                break
        if not valid: continue
        # 4th: x-3d, x-2d, x-d, x
        for d in range(1, x // 3 + 1):
            if ((1 << (x - 3*d)) & selected_mask) and ((1 << (x - 2*d)) & selected_mask) and ((1 << (x - d)) & selected_mask):
                valid = False
                break
        if valid:
            selected_mask |= x_bit
    return selected_mask

def _count_3aps(mask, aps3, aps3_other_masks, elem_to_aps3):
    """Count 3-APs in a set represented by bitmask."""
    count = 0
    for i, (a, b, c) in enumerate(aps3):
        if (mask & (1 << a)) and (mask & (1 << b)) and (mask & (1 << c)):
            count += 1
    return count

def _local_search(start_mask, start_score, aps3, aps4, aps3_other_masks, aps4_other_masks, elem_to_aps3, elem_to_aps4, deadline):
    """Local search via swaps using bitmask operations."""
    mask = start_mask
    score = start_score
    best_mask = mask
    best_score = score
    all_mask = (1 << (N + 1)) - 2  # bits 1..N set
    
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        # Get list of elements in A
        a_elements = [x for x in range(1, N + 1) if (mask >> x) & 1]
        # Get list of elements not in A
        not_a_elements = [x for x in range(1, N + 1) if not ((mask >> x) & 1)]
        
        random.shuffle(a_elements)
        random.shuffle(not_a_elements)
        
        for x in a_elements:
            if time.time() > deadline:
                break
            x_bit = 1 << x
            mask_without_x = mask & ~x_bit
            
            for y in not_a_elements:
                if time.time() > deadline:
                    break
                y_bit = 1 << y
                
                # Check validity: no new 4-AP with y added (after removing x)
                valid = True
                for (ap_idx, pos) in elem_to_aps4[y]:
                    other_mask = aps4_other_masks[ap_idx][pos]
                    # Check if other 3 elements are all in mask_without_x
                    if (mask_without_x & other_mask) == other_mask:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Calculate score delta
                # Loss: 3-APs containing x where other 2 are in mask
                loss = 0
                for (ap_idx, pos) in elem_to_aps3[x]:
                    other_mask = aps3_other_masks[ap_idx][pos]
                    if (mask & other_mask) == other_mask:
                        loss += 1
                
                # Gain: 3-APs containing y where other 2 are in mask_without_x
                gain = 0
                for (ap_idx, pos) in elem_to_aps3[y]:
                    other_mask = aps3_other_masks[ap_idx][pos]
                    if (mask_without_x & other_mask) == other_mask:
                        gain += 1
                
                delta = gain - loss
                if delta > 0:
                    # Apply swap
                    mask = mask_without_x | y_bit
                    score += delta
                    improved = True
                    if score > best_score:
                        best_score = score
                        best_mask = mask
                    break  # Restart with new state
    
    return best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    aps3, aps4, aps3_other_masks, aps4_other_masks, elem_to_aps3, elem_to_aps4 = _precompute()
    
    # Initial greedy set
    init_mask = _initial_greedy_set()
    init_score = _count_3aps(init_mask, aps3, aps3_other_masks, elem_to_aps3)
    
    best_mask = init_mask
    best_score = init_score
    
    # Local search
    best_mask, best_score = _local_search(init_mask, init_score, aps3, aps4, 
                                          aps3_other_masks, aps4_other_masks,
                                          elem_to_aps3, elem_to_aps4, deadline - 10)
    
    # If time allows, try a few random restarts with perturbations
    if time.time() < deadline - 5:
        for restart in range(20):
            if time.time() > deadline - 3:
                break
            
            # Start from best solution and perturb
            perturbed_mask = best_mask
            # Remove 3-5 random elements and add different ones
            a_elements = [x for x in range(1, N + 1) if (perturbed_mask >> x) & 1]
            not_a_elements = [x for x in range(1, N + 1) if not ((perturbed_mask >> x) & 1)]
            
            random.shuffle(a_elements)
            random.shuffle(not_a_elements)
            
            num_remove = min(3 + restart % 3, len(a_elements))
            for i in range(num_remove):
                if not_a_elements:
                    x = a_elements[i]
                    y = random.choice(not_a_elements)
                    perturbed_mask &= ~(1 << x)
                    perturbed_mask |= (1 << y)
            
            perturbed_score = _count_3aps(perturbed_mask, aps3, aps3_other_masks, elem_to_aps3)
            
            # Local search from perturbed state
            search_deadline = min(deadline - 3, time.time() + 15)
            new_mask, new_score = _local_search(perturbed_mask, perturbed_score, aps3, aps4,
                                                aps3_other_masks, aps4_other_masks,
                                                elem_to_aps3, elem_to_aps4, search_deadline)
            
            if new_score > best_score:
                best_score = new_score
                best_mask = new_mask
    
    # Convert bitmask to sorted list
    result = [x for x in range(1, N + 1) if (best_mask >> x) & 1]
    return result

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We need to ensure the entry point is exactly 'construct'
# The code above defines it.
# EVOLVE-BLOCK-END
