import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
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
    # For each element, precompute which 4-APs it's in and the bitmask of other elements in those APs
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
        # When we remove x and add y, we need to check y's 4-APs
        # The set after swap is (A - {x}) ∪ {y}
        # For each 4-AP containing y, count how many of the other 3 are in (A - {x})
        for mask in elem_4ap_bitmasks[y]:
            # mask contains the other 3 elements in the 4-AP
            # We need to count how many of these are in A but not x
            count = (A_mask & mask).bit_count()
            # If x is in mask, we need to subtract it
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
            # Check if both a and b are in A after removing x
            in_a = in_A(a) and a != x
            in_b = in_A(b) and b != x
            if in_a and in_b:
                gain += 1
        
        return gain - loss
    
    best_set_mask = 0
    best_score = 0
    
    # Phase 0: Initialization
    # 0a: Deterministic Greedy
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
    
    # 0b: Randomized Greedy Restarts
    for restart in range(8):
        if time.time() > deadline - 30.0: # Reserve time for SA
            break
        A_mask = 0
        current_3ap_count = 0
        remaining = set(range(1, N + 1))
        
        while remaining:
            best_x = None
            best_gain = -1
            candidates = list(remaining)
            random.shuffle(candidates)
            # Sample top 50 to speed up initialization
            for x in candidates[:50]:
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
    
    # Phase 1: Simulated Annealing
    # Use the improved cooling schedule from the inspiration
    T_start = 80.0
    T_end = 0.1
    total_time = deadline - start_time
    # We want to run SA for most of the time, leaving ~5s for polish
    sa_end = start_time + total_time * 0.9
    
    while time.time() < sa_end:
        elapsed = time.time() - start_time
        progress = min(elapsed / (sa_end - start_time), 1.0)
        # Exponential cooling
        T = T_start * np.exp(-5.0 * progress)
        T = max(T, T_end)
        
        r = random.random()
        
        if r < 0.25:
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
        elif r < 0.4:
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
    
    # Phase 2: Exhaustive Local Search Polish
    # This phase is crucial for final improvements
    A_mask = best_set_mask
    current_3ap_count = best_score
    improved = True
    polish_rounds = 0
    max_polish_rounds = 20
    
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
        
        # Try ALL swaps
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
        
        if current_3ap_count > best_score:
            best_score = current_3ap_count
            best_set_mask = A_mask

    # Extract result
    result = [x for x in range(1, N + 1) if in_A(x)]
    return sorted(result)
