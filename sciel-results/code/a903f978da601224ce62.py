import time
import random
import math

N = 256
DEADLINE = 85.0  # 90s limit, keep 5s safety margin

def _precompute():
    """
    Precompute 3-APs and 4-APs.
    """
    aps3 = []
    aps4 = []
    elem_to_aps3 = [[] for _ in range(N + 1)]
    elem_to_aps4 = [[] for _ in range(N + 1)]
    
    # Generate 3-APs: (a, a+d, a+2d)
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            aps3.append((a, b, c))
            mask_3 = (1 << a) | (1 << b) | (1 << c)
            elem_to_aps3[a].append((1 << b) | (1 << c))
            elem_to_aps3[b].append((1 << a) | (1 << c))
            elem_to_aps3[c].append((1 << a) | (1 << b))
            
    # Generate 4-APs: (a, a+d, a+2d, a+3d)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps4.append((a, b, c, e))
            elem_to_aps4[a].append((1 << b) | (1 << c) | (1 << e))
            elem_to_aps4[b].append((1 << a) | (1 << c) | (1 << e))
            elem_to_aps4[c].append((1 << a) | (1 << b) | (1 << e))
            elem_to_aps4[e].append((1 << a) | (1 << b) | (1 << c))
    
    return aps3, aps4, elem_to_aps3, elem_to_aps4

def _initial_greedy_mask(elem_to_aps3, elem_to_aps4, random_order=False):
    """
    Greedy construction: add x if it doesn't complete a 4-AP.
    """
    A_mask = 0
    ap_count = [len(elem_to_aps3[i]) for i in range(N + 1)]
    
    if random_order:
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
    else:
        candidates = sorted(range(1, N + 1), key=lambda x: -ap_count[x])
        
    for x in candidates:
        valid = True
        for mask_other3 in elem_to_aps4[x]:
            if (A_mask & mask_other3) == mask_other3:
                valid = False
                break
        if valid:
            A_mask |= (1 << x)
    return A_mask

def _count_3aps(mask, aps3):
    count = 0
    for a, b, c in aps3:
        if (mask & (1 << a)) and (mask & (1 << b)) and (mask & (1 << c)):
            count += 1
    return count

def _local_search(mask, aps3, aps4, elem_to_aps3, elem_to_aps4, deadline):
    """
    Best-improvement local search.
    """
    current_score = _count_3aps(mask, aps3)
    in_set = [i for i in range(1, N + 1) if (mask >> i) & 1]
    out_set = [i for i in range(1, N + 1) if not ((mask >> i) & 1)]
    
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        best_delta = 0
        best_x = -1
        best_y = -1
        
        # Iterate all possible swaps
        for x in in_set:
            for y in out_set:
                # Check validity: does adding y create a 4-AP?
                valid = True
                for mask_other3 in elem_to_aps4[y]:
                    # The other 3 elements must be in (mask - {x})
                    # This is equivalent to: (mask & mask_other3) == mask_other3 
                    # AND x is not in mask_other3 (because if x is in mask_other3, 
                    # removing x breaks the AP, so it's valid).
                    # Wait, if x is in mask_other3, then (mask & mask_other3) == mask_other3 
                    # implies x is in mask. Since we remove x, the AP is broken.
                    # If x is NOT in mask_other3, and (mask & mask_other3) == mask_other3,
                    # then the AP is formed by elements in mask, none of which is x, 
                    # so it remains in mask - {x}. Invalid.
                    if (mask & mask_other3) == mask_other3:
                        if (mask_other3 & (1 << x)) == 0:
                            valid = False
                            break
                            
                if not valid:
                    continue
                    
                # Calculate delta
                # Loss: 3-APs with x where other 2 are in mask
                loss = 0
                for mask_other2 in elem_to_aps3[x]:
                    if (mask & mask_other2) == mask_other2:
                        loss += 1
                        
                # Gain: 3-APs with y where other 2 are in (mask - {x})
                gain = 0
                for mask_other2 in elem_to_aps3[y]:
                    # Other 2 must be in mask, and x must NOT be one of them
                    if (mask & mask_other2) == mask_other2:
                        if (mask_other2 & (1 << x)) == 0:
                            gain += 1
                            
                delta = gain - loss
                if delta > best_delta:
                    best_delta = delta
                    best_x = x
                    best_y = y
                    
        if best_delta > 0:
            # Apply swap
            mask &= ~(1 << best_x)
            mask |= (1 << best_y)
            current_score += best_delta
            improved = True
            
            # Update in_set and out_set
            in_set.remove(best_x)
            out_set.remove(best_y)
            in_set.append(best_y)
            
    return mask, current_score

def construct():
    start_time = time.time()
    aps3, aps4, elem_to_aps3, elem_to_aps4 = _precompute()
    
    best_mask = 0
    best_score = 0
    
    # Multi-start local search
    # Start 1: Greedy
    mask = _initial_greedy_mask(elem_to_aps3, elem_to_aps4, random_order=False)
    mask, score = _local_search(mask, aps3, aps4, elem_to_aps3, elem_to_aps4, DEADLINE)
    if score > best_score:
        best_score = score
        best_mask = mask
        
    # Start 2: Random Greedy
    if time.time() < DEADLINE:
        mask = _initial_greedy_mask(elem_to_aps3, elem_to_aps4, random_order=True)
        mask, score = _local_search(mask, aps3, aps4, elem_to_aps3, elem_to_aps4, DEADLINE)
        if score > best_score:
            best_score = score
            best_mask = mask
            
    # Start 3+: Perturbation of best
    while time.time() < DEADLINE - 1.0:
        # Create a perturbed version of best_mask
        # Remove k random elements and add k random valid elements?
        # Or just swap k random pairs?
        # Let's do a random kick: swap 5 random pairs.
        current_mask = best_mask
        in_set = [i for i in range(1, N + 1) if (current_mask >> i) & 1]
        out_set = [i for i in range(1, N + 1) if not ((current_mask >> i) & 1)]
        
        if len(in_set) < 5 or len(out_set) < 5:
            break
            
        # Perform 5 random swaps (ignoring validity for the kick, 
        # local search will fix it or reject if it can't recover? 
        # Actually, if we create an invalid set, local search (which assumes validity) 
        # might behave incorrectly or get stuck.
        # Better: only kick with valid swaps? That's just local search.
        # Alternative kick: Remove 5 random elements, then greedily add back 5 best valid elements.
        
        # Let's try: Remove 5 random elements.
        kick_in = random.sample(in_set, 5)
        for x in kick_in:
            current_mask &= ~(1 << x)
            
        # Now greedily add back 5 elements that maximize 3-APs without creating 4-APs
        # We can just run a few steps of greedy addition
        candidates = sorted(range(1, N + 1), key=lambda x: -len(elem_to_aps3[x]))
        added = 0
        for y in candidates:
            if (current_mask >> y) & 1:
                continue
            valid = True
            for mask_other3 in elem_to_aps4[y]:
                if (current_mask & mask_other3) == mask_other3:
                    valid = False
                    break
            if valid:
                current_mask |= (1 << y)
                added += 1
                if added >= 5:
                    break
                    
        mask, score = _local_search(current_mask, aps3, aps4, elem_to_aps3, elem_to_aps4, DEADLINE)
        if score > best_score:
            best_score = score
            best_mask = mask
            
    return sorted([i for i in range(1, N + 1) if (best_mask >> i) & 1])
