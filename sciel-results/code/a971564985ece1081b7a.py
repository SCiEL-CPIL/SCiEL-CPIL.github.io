import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 52.0  # Safety margin for 60s limit
    N = 256
    
    # Precompute 4-APs and 3-APs
    all_4aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            all_4aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    all_3aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            all_3aps.append((a, a + d, a + 2 * d))
    
    # Map element to list of AP indices it belongs to
    elem_to_4aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(all_4aps):
        for x in ap:
            elem_to_4aps[x].append(idx)
    
    elem_to_3aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(all_3aps):
        for x in ap:
            elem_to_3aps[x].append(idx)
    
    # Precompute bitmask representations for faster validity checks
    elem_4ap_bitmasks = {}
    for i in range(1, N + 1):
        masks = []
        for idx in elem_to_4aps[i]:
            ap = all_4aps[idx]
            mask = 0
            for y in ap:
                if y != i:
                    mask |= (1 << (y - 1))
            masks.append(mask)
        elem_4ap_bitmasks[i] = masks
    
    # For 3-APs, precompute pairs that would complete a 3-AP
    elem_3ap_pairs = {}
    for i in range(1, N + 1):
        pairs = []
        for idx in elem_to_3aps[i]:
            ap = all_3aps[idx]
            others = [y for y in ap if y != i]
            if len(others) == 2:
                pairs.append((others[0], others[1]))
        elem_3ap_pairs[i] = pairs
    
    # Precompute for each 4-AP the bitmask of all 4 elements
    all_4ap_masks = []
    for ap in all_4aps:
        mask = 0
        for x in ap:
            mask |= (1 << (x - 1))
        all_4ap_masks.append(mask)
    
    # Use integer bitmask for fast operations
    A_mask = 0
    
    def in_A(x):
        return (A_mask >> (x-1)) & 1
    
    def add_to_A(x):
        nonlocal A_mask
        A_mask |= (1 << (x-1))
    
    def remove_from_A(x):
        nonlocal A_mask
        A_mask &= ~(1 << (x-1))
    
    def is_valid_add(x):
        """Check if adding x keeps set 4-AP free (x not in A)"""
        for mask in elem_4ap_bitmasks[x]:
            if (A_mask & mask).bit_count() >= 3:
                return False
        return True
    
    def is_valid_swap(x, y):
        """Check if swapping x out and y in keeps set 4-AP free"""
        for mask in elem_4ap_bitmasks[y]:
            count = (A_mask & mask).bit_count()
            if (mask >> (x-1)) & 1:
                count -= 1
            if count >= 3:
                return False
        return True
    
    def gain_add(x):
        """Calculate gain in 3-AP count if we add x to A"""
        gain = 0
        for (y, z) in elem_3ap_pairs[x]:
            if in_A(y) and in_A(z):
                gain += 1
        return gain
    
    def loss_remove(x):
        """Calculate loss in 3-AP count if we remove x from A"""
        loss = 0
        for (y, z) in elem_3ap_pairs[x]:
            if in_A(y) and in_A(z):
                loss += 1
        return loss
    
    def delta_swap(x, y):
        """Calculate change in 3-AP count if we swap x (out) and y (in)"""
        loss = loss_remove(x)
        gain = 0
        for (a, b) in elem_3ap_pairs[y]:
            in_a = in_A(a) and a != x
            in_b = in_A(b) and b != x
            if in_a and in_b:
                gain += 1
        return gain - loss
    
    def compute_3ap_count():
        """Compute total number of 3-APs in current A"""
        count = 0
        for idx, ap in enumerate(all_3aps):
            a, b, c = ap
            if in_A(a) and in_A(b) and in_A(c):
                count += 1
        return count
    
    # Greedy initialization: deterministic first pass, then randomized restarts
    best_set_mask = 0
    best_score = 0
    
    # Deterministic greedy: always pick the element with max gain
    A_mask = 0
    current_3ap_count = 0
    remaining = set(range(1, N + 1))
    while remaining:
        best_x = None
        best_gain = -1
        for x in sorted(remaining):
            if is_valid_add(x):
                g = gain_add(x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
        if best_x is None:
            break
        add_to_A(best_x)
        current_3ap_count += best_gain
        remaining.remove(best_x)
    if current_3ap_count > best_score:
        best_score = current_3ap_count
        best_set_mask = A_mask
    
    # Randomized greedy restarts
    for restart in range(12):
        A_mask = 0
        current_3ap_count = 0
        remaining = set(range(1, N + 1))
        
        while remaining:
            best_x = None
            best_gain = -1
            candidates = list(remaining)
            random.shuffle(candidates)
            for x in candidates[:60]:
                if is_valid_add(x):
                    g = gain_add(x)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
            if best_x is None:
                break
            add_to_A(best_x)
            current_3ap_count += best_gain
            remaining.remove(best_x)
        
        if current_3ap_count > best_score:
            best_score = current_3ap_count
            best_set_mask = A_mask
    
    # Restore best state
    A_mask = best_set_mask
    current_3ap_count = best_score
    
    # Phase 1: Max-SAT-inspired local search with clause-weight bias
    # Hard clauses: 4-APs (must not be satisfied)
    # Soft clauses: 3-APs (reward satisfaction)
    # Use clause-weight updates to bias search toward "hard" regions
    T_start = 60.0
    T_end = 0.05
    total_time = deadline - start_time
    phase1_end = start_time + total_time * 0.55
    
    # Precompute for each element: how many 4-APs it's in (for weighting)
    elem_4ap_count = {i: len(elem_to_4aps[i]) for i in range(1, N + 1)}
    
    while time.time() < phase1_end:
        elapsed = time.time() - start_time
        progress = min(elapsed / (phase1_end - start_time), 1.0)
        T = T_start * np.exp(-4.0 * progress)
        T = max(T, T_end)
        
        r = random.random()
        
        if r < 0.20:
            # Try Add - biased toward elements that complete more 3-APs
            x = random.randint(1, N)
            if not in_A(x):
                if is_valid_add(x):
                    delta = gain_add(x)
                    # Max-SAT clause weight: elements in more 4-APs are "riskier"
                    # but also potentially more rewarding
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        add_to_A(x)
                        current_3ap_count += delta
                        if current_3ap_count > best_score:
                            best_score = current_3ap_count
                            best_set_mask = A_mask
        elif r < 0.35:
            # Try Remove - biased toward elements in many 4-APs (harder constraints)
            x = random.randint(1, N)
            if in_A(x):
                delta = -loss_remove(x)
                # Weight by 4-AP count: removing elements in many 4-APs is more "valuable"
                weight = 1.0 + 0.1 * elem_4ap_count[x]
                if delta >= 0 or random.random() < np.exp(delta / (T * weight)):
                    remove_from_A(x)
                    current_3ap_count += delta
                    if current_3ap_count > best_score:
                        best_score = current_3ap_count
                        best_set_mask = A_mask
        elif r < 0.75:
            # Try Swap - standard
            x = random.randint(1, N)
            y = random.randint(1, N)
            if in_A(x) and not in_A(y):
                if is_valid_swap(x, y):
                    delta = delta_swap(x, y)
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        remove_from_A(x)
                        add_to_A(y)
                        current_3ap_count += delta
                        if current_3ap_count > best_score:
                            best_score = current_3ap_count
                            best_set_mask = A_mask
        else:
            # Try 2-swap: remove two, add two (more complex moves)
            x1 = random.randint(1, N)
            x2 = random.randint(1, N)
            y1 = random.randint(1, N)
            y2 = random.randint(1, N)
            if (in_A(x1) and in_A(x2) and not in_A(y1) and not in_A(y2)
                    and x1 != x2 and y1 != y2 and x1 != y1 and x1 != y2 and x2 != y1 and x2 != y2):
                # Check validity: remove x1,x2 add y1,y2
                # Temporarily remove x1, x2
                A_mask_saved = A_mask
                remove_from_A(x1)
                remove_from_A(x2)
                
                valid = True
                for mask in elem_4ap_bitmasks[y1]:
                    if (A_mask & mask).bit_count() >= 3:
                        valid = False
                        break
                if valid:
                    add_to_A(y1)
                    for mask in elem_4ap_bitmasks[y2]:
                        if (A_mask & mask).bit_count() >= 3:
                            valid = False
                            break
                    if valid:
                        add_to_A(y2)
                
                if valid:
                    # Compute delta
                    delta = compute_3ap_count() - current_3ap_count
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        current_3ap_count += delta
                        if current_3ap_count > best_score:
                            best_score = current_3ap_count
                            best_set_mask = A_mask
                    else:
                        # Revert
                        A_mask = A_mask_saved
                else:
                    # Revert
                    A_mask = A_mask_saved
    
    # Phase 2: Standard SA with faster exploitation
    A_mask = best_set_mask
    current_3ap_count = best_score
    phase2_end = start_time + total_time * 0.85
    
    T_start2 = 30.0
    T_end2 = 0.02
    
    while time.time() < phase2_end:
        elapsed = time.time() - start_time
        progress = min((elapsed - (phase1_end - start_time)) / (phase2_end - phase1_end), 1.0)
        T = T_start2 * np.exp(-4.0 * max(progress, 0))
        T = max(T, T_end2)
        
        r = random.random()
        
        if r < 0.15:
            # Try Add
            x = random.randint(1, N)
            if not in_A(x):
                if is_valid_add(x):
                    delta = gain_add(x)
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        add_to_A(x)
                        current_3ap_count += delta
                        if current_3ap_count > best_score:
                            best_score = current_3ap_count
                            best_set_mask = A_mask
        elif r < 0.30:
            # Try Remove
            x = random.randint(1, N)
            if in_A(x):
                delta = -loss_remove(x)
                if delta >= 0 or random.random() < np.exp(delta / T):
                    remove_from_A(x)
                    current_3ap_count += delta
                    if current_3ap_count > best_score:
                        best_score = current_3ap_count
                        best_set_mask = A_mask
        else:
            # Try Swap
            x = random.randint(1, N)
            y = random.randint(1, N)
            if in_A(x) and not in_A(y):
                if is_valid_swap(x, y):
                    delta = delta_swap(x, y)
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        remove_from_A(x)
                        add_to_A(y)
                        current_3ap_count += delta
                        if current_3ap_count > best_score:
                            best_score = current_3ap_count
                            best_set_mask = A_mask
    
    # Phase 3: Exhaustive local search polish
    A_mask = best_set_mask
    current_3ap_count = best_score
    improved = True
    polish_rounds = 0
    max_polish_rounds = 15
    
    while improved and time.time() < deadline and polish_rounds < max_polish_rounds:
        improved = False
        polish_rounds += 1
        
        # Try adding ALL elements
        for x in range(1, N + 1):
            if not in_A(x):
                if is_valid_add(x):
                    delta = gain_add(x)
                    if delta > 0:
                        add_to_A(x)
                        current_3ap_count += delta
                        improved = True
        
        # Try removing ALL elements
        for x in range(1, N + 1):
            if in_A(x):
                delta = -loss_remove(x)
                if delta > 0:
                    remove_from_A(x)
                    current_3ap_count += delta
                    improved = True
        
        # Try ALL swaps (full search, not sampled)
        active = [x for x in range(1, N + 1) if in_A(x)]
        inactive = [x for x in range(1, N + 1) if not in_A(x)]
        
        for x in active:
            for y in inactive:
                if is_valid_swap(x, y):
                    delta = delta_swap(x, y)
                    if delta > 0:
                        remove_from_A(x)
                        add_to_A(y)
                        current_3ap_count += delta
                        improved = True
    
    # Extract result
    result = [x for x in range(1, N + 1) if in_A(x)]
    return sorted(result)
