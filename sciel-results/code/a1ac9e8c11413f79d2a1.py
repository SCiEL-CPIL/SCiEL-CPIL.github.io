import random
import time
import math

# Precompute 3-APs and 4-APs
# We represent elements as 0-indexed for bitmasks, but 1-indexed for logic if needed.
# Let's stick to 0-indexed internally: 0..255 corresponding to 1..256.

def get_aps(n, k):
    """
    Get all k-term arithmetic progressions in [0, n-1].
    Returns a list of tuples of indices.
    """
    aps = []
    max_d = (n - 1) // (k - 1)
    for d in range(1, max_d + 1):
        for start in range(0, n - (k - 1) * d):
            ap = tuple(start + i * d for i in range(k))
            aps.append(ap)
    return aps

# Precomputation
N = 256
AP3_LIST = get_aps(N, 3)
AP4_LIST = get_aps(N, 4)

# Convert to bitmasks
AP3_MASKS = []
for ap in AP3_LIST:
    mask = 0
    for idx in ap:
        mask |= (1 << idx)
    AP3_MASKS.append(mask)

AP4_MASKS = []
for ap in AP4_LIST:
    mask = 0
    for idx in ap:
        mask |= (1 << idx)
    AP4_MASKS.append(mask)

# Index APs by element for fast local checking
# ap4_by_index[i] = list of indices in AP4_LIST that contain element i
ap4_by_index = [[] for _ in range(N)]
for i, ap in enumerate(AP4_LIST):
    for idx in ap:
        ap4_by_index[idx].append(i)

# ap3_by_index[i] = list of indices in AP3_LIST that contain element i
# Used for score calculation updates
ap3_by_index = [[] for _ in range(N)]
for i, ap in enumerate(AP3_LIST):
    for idx in ap:
        ap3_by_index[idx].append(i)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Initialize with greedy set
    selected = set()
    for x in range(N):
        # Check if adding x creates a 4-AP
        valid = True
        for ap_idx in ap4_by_index[x]:
            ap = AP4_LIST[ap_idx]
            # Check if other elements are in selected
            others_in = all((other in selected) for other in ap if other != x)
            if others_in:
                valid = False
                break
        if valid:
            selected.add(x)
    
    # Convert to bitmask for fast operations
    current_mask = 0
    for x in selected:
        current_mask |= (1 << x)
        
    # Calculate initial score
    current_score = 0
    for mask in AP3_MASKS:
        if (current_mask & mask) == mask:
            current_score += 1
            
    # Check initial violations (should be 0)
    current_violations = 0
    for mask in AP4_MASKS:
        if (current_mask & mask) == mask:
            current_violations += 1

    best_mask = current_mask
    best_score = current_score
    best_violations = current_violations

    # Simulated Annealing Parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # Helper to calculate score and violations for a given mask
    # This is expensive, so we try to update incrementally if possible, 
    # but for simplicity and correctness, we can recalculate if changes are small?
    # Actually, recalculating full score is ~15k ops. 
    # If we do 10k steps, that's 1.5e8 ops. Might be tight.
    # Let's try incremental updates.
    
    # State:
    # current_mask: int
    # current_score: int
    # current_violations: int
    
    # To update score when adding/removing x:
    # Delta score = (number of 3-APs containing x that are now fully in set) - (number of 3-APs containing x that were fully in set but are no longer)
    # This requires knowing the state of other elements in those APs.
    
    # Let's define a helper to compute the contribution of an element to the score/violations
    # given the current mask.
    
    def calc_delta_add(x, mask):
        """
        Returns (delta_score, delta_violations) if we add x to the set represented by mask.
        Assumes x is not in mask.
        """
        delta_s = 0
        delta_v = 0
        
        # Check 3-APs containing x
        # A 3-AP becomes complete if the other 2 elements are in mask.
        for ap_idx in ap3_by_index[x]:
            ap = AP3_LIST[ap_idx]
            # Find the other two elements
            other1 = -1
            other2 = -1
            count = 0
            for idx in ap:
                if idx != x:
                    if count == 0:
                        other1 = idx
                    else:
                        other2 = idx
                    count += 1
            
            if (mask & (1 << other1)) and (mask & (1 << other2)):
                delta_s += 1
        
        # Check 4-APs containing x
        # A 4-AP becomes a violation if the other 3 elements are in mask.
        for ap_idx in ap4_by_index[x]:
            ap = AP4_LIST[ap_idx]
            others_in = True
            for idx in ap:
                if idx != x:
                    if not (mask & (1 << idx)):
                        others_in = False
                        break
            if others_in:
                delta_v += 1
                
        return delta_s, delta_v

    def calc_delta_remove(x, mask):
        """
        Returns (delta_score, delta_violations) if we remove x from the set represented by mask.
        Assumes x is in mask.
        """
        delta_s = 0
        delta_v = 0
        
        # Check 3-APs containing x
        # A 3-AP stops being complete if x was the last missing piece? No, x is present.
        # If we remove x, any 3-AP containing x that was previously complete becomes incomplete.
        for ap_idx in ap3_by_index[x]:
            ap = AP3_LIST[ap_idx]
            others_in = True
            for idx in ap:
                if idx != x:
                    if not (mask & (1 << idx)):
                        others_in = False
                        break
            if others_in:
                # The AP was complete (x + others). Now it's not.
                delta_s -= 1
        
        # Check 4-APs containing x
        # If we remove x, any 4-AP containing x that was previously a violation stops being one.
        for ap_idx in ap4_by_index[x]:
            ap = AP4_LIST[ap_idx]
            others_in = True
            for idx in ap:
                if idx != x:
                    if not (mask & (1 << idx)):
                        others_in = False
                        break
            if others_in:
                # The AP was a violation (x + others). Now it's not.
                delta_v -= 1
                
        return delta_s, delta_v

    # Main SA Loop
    while T > min_T and time.time() < deadline:
        # Choose a move: Add, Remove, or Swap
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            # Pick a random x not in current_mask
            available = [i for i in range(N) if not (current_mask & (1 << i))]
            if not available:
                continue
            x = random.choice(available)
            
            new_mask = current_mask | (1 << x)
            ds, dv = calc_delta_add(x, current_mask)
            
            new_score = current_score + ds
            new_violations = current_violations + dv
            
            # Penalty for violations
            # Objective: Maximize Score, Minimize Violations
            # Energy E = -Score + Large_Penalty * Violations
            # Delta E = -Delta Score + Large_Penalty * Delta Violations
            penalty_weight = 100.0
            delta_e = -ds + penalty_weight * dv
            
            # Acceptance criterion
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                current_mask = new_mask
                current_score = new_score
                current_violations = new_violations
                
                if current_violations == 0 and current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    
        elif move_type == 'remove':
            # Pick a random x in current_mask
            present = [i for i in range(N) if (current_mask & (1 << i))]
            if not present:
                continue
            x = random.choice(present)
            
            new_mask = current_mask & ~(1 << x)
            ds, dv = calc_delta_remove(x, current_mask)
            
            new_score = current_score + ds
            new_violations = current_violations + dv
            
            penalty_weight = 100.0
            delta_e = -ds + penalty_weight * dv
            
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                current_mask = new_mask
                current_score = new_score
                current_violations = new_violations
                
                if current_violations == 0 and current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask

        else: # swap
            # Pick x in, y not in
            present = [i for i in range(N) if (current_mask & (1 << i))]
            available = [i for i in range(N) if not (current_mask & (1 << i))]
            if not present or not available:
                continue
            x = random.choice(present)
            y = random.choice(available)
            
            # Remove x, Add y
            # Order matters for delta calculation?
            # Let's calculate delta for removing x first, then adding y to the intermediate state.
            # Or just compute the new state's metrics directly?
            # Direct computation is safer for swaps to avoid intermediate state errors.
            # But direct computation is O(N) for score? No, O(degree).
            # Let's do incremental:
            # 1. Remove x from current_mask -> mask_mid
            # 2. Add y to mask_mid -> new_mask
            
            mask_mid = current_mask & ~(1 << x)
            ds1, dv1 = calc_delta_remove(x, current_mask)
            
            # Now add y to mask_mid
            ds2, dv2 = calc_delta_add(y, mask_mid)
            
            new_score = current_score + ds1 + ds2
            new_violations = current_violations + dv1 + dv2
            new_mask = mask_mid | (1 << y)
            
            penalty_weight = 100.0
            delta_e = -(ds1 + ds2) + penalty_weight * (dv1 + dv2)
            
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                current_mask = new_mask
                current_score = new_score
                current_violations = new_violations
                
                if current_violations == 0 and current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask

        T *= cooling

    # Final check: ensure best_mask is valid
    # If best_mask has violations, we need to repair it or return the best valid one found.
    # The SA might end in a state with violations. We track best_score only when violations==0.
    # However, if no valid state was found (unlikely given greedy start), we should repair.
    
    # Repair best_mask if it has violations (shouldn't happen if we only update best when valid)
    # But what if the initial greedy set had violations? (It doesn't by construction).
    # What if we never found a valid state better than greedy? Then best is greedy.
    
    # Convert best_mask back to sorted list of 1-based integers
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
