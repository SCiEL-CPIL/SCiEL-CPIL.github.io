import time
import random

def construct():
    start_time = time.time()
    # Safety margin: The limit is 60s. We stop at 55s to ensure return.
    deadline = start_time + 55.0
    
    n = 512
    
    # Precompute all 4-APs and their bitmasks
    # ap_list: list of tuples (a, b, c, d)
    # ap_masks: list of int bitmasks
    # elem_to_ap_indices: list of lists of AP indices for each element 1..n
    ap_list = []
    ap_masks = []
    elem_to_ap_indices = [[] for _ in range(n + 1)]
    
    # Generate 4-APs: a, a+d, a+2d, a+3d
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > n:
                continue
            ap = (a, b, c, e)
            ap_idx = len(ap_list)
            ap_list.append(ap)
            
            # Create bitmask
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            ap_masks.append(mask)
            
            # Map elements to AP indices
            elem_to_ap_indices[a].append(ap_idx)
            elem_to_ap_indices[b].append(ap_idx)
            elem_to_ap_indices[c].append(ap_idx)
            elem_to_ap_indices[e].append(ap_idx)
            
    num_aps = len(ap_list)
    
    # Precompute "other masks" for fast blocking checks
    # other_masks[ap_idx][i] where i is 0..3 corresponding to position in ap_tuple
    # Actually, easier: for each element x, we need to know which APs block it.
    # An AP blocks x if the other 3 elements are in S.
    # Let's precompute for each element x, a list of (ap_idx, mask_of_other_3)
    # This allows: if (S_mask & mask_of_other_3) == mask_of_other_3, then blocked.
    
    elem_blockers = [[] for _ in range(n + 1)]
    for ap_idx, ap in enumerate(ap_list):
        mask = ap_masks[ap_idx]
        for x in ap:
            # Mask of other 3 elements
            x_bit = 1 << (x - 1)
            other_mask = mask & ~x_bit
            elem_blockers[x].append((ap_idx, other_mask))
            
    def can_add(x, S_mask):
        """
        Check if x can be added to S_mask without forming a 4-AP.
        Returns True if valid.
        """
        for _, other_mask in elem_blockers[x]:
            if (S_mask & other_mask) == other_mask:
                return False
        return True

    def get_blocking_aps(S_mask):
        """
        Return a list of indices of APs that have exactly 3 elements in S_mask.
        """
        blocking = []
        # Optimization: We can iterate through all APs. 512 elements -> ~few thousand APs.
        # For each AP, count bits in S_mask & ap_mask.
        # If count == 3, it's blocking.
        # This is O(num_aps * bit_count). bit_count is fast.
        for i in range(num_aps):
            m = ap_masks[i]
            # Count bits set in intersection
            if bin(S_mask & m).count('1') == 3:
                blocking.append(i)
        return blocking

    best_size = 0
    best_mask = 0
    best_set = set()
    
    # Multi-start local search
    # We use a fixed number of restarts if time permits, or loop until deadline.
    # Given the speed of bitmask ops, we can do many restarts.
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        if time.time() > deadline:
            break
            
        # 1. Greedy Construction
        S_mask = 0
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S_mask):
                S_mask |= (1 << (x - 1))
                
        current_size = bin(S_mask).count('1')
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
            best_set = {i+1 for i in range(n) if (S_mask >> i) & 1}
            
        # 2. Local Search Phase
        # We alternate between trying to add elements and performing swaps.
        # To avoid getting stuck, we limit the number of local search steps per restart.
        # But we also want to run as long as possible if it's improving or stagnating.
        
        local_steps = 0
        max_local_steps = 20000 # High limit, but time is the real constraint
        no_improve_counter = 0
        max_no_improve = 500 # Restart if stuck for a while
        
        while local_steps < max_local_steps and time.time() < deadline:
            local_steps += 1
            if time.time() > deadline:
                break
                
            improved = False
            
            # A. Try to add any element
            # To make this efficient, we can iterate through elements not in S.
            # But checking all 512 every step is O(512 * num_aps_per_elem).
            # Let's try a random subset first, or just iterate all if fast enough.
            # For N=512, iterating all is ~512 * ~10 checks = 5000 ops. Very fast.
            
            # We maintain a list of candidates not in S for efficiency?
            # Or just iterate 1..n.
            # Let's iterate 1..n to ensure we don't miss easy additions.
            
            added_any = False
            # Optimization: Only check elements not in S
            # We can generate list of non-selected elements
            # But creating a list every time is expensive.
            # Let's just iterate and check bit.
            
            for x in range(1, n + 1):
                if not (S_mask & (1 << (x - 1))):
                    if can_add(x, S_mask):
                        S_mask |= (1 << (x - 1))
                        current_size += 1
                        improved = True
                        added_any = True
                        # Don't break, try to add more in this pass
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = S_mask
                            best_set = {i+1 for i in range(n) if (S_mask >> i) & 1}
                            
            if added_any:
                no_improve_counter = 0
                continue # Restart loop to check for more additions or swaps
                
            # B. No additions possible. Try swaps.
            # Find blocking APs
            blocking_aps = get_blocking_aps(S_mask)
            
            if not blocking_aps:
                # If no blocking APs, why can't we add?
                # This implies that for every x not in S, there is an AP with 3 elements in S.
                # So there MUST be blocking APs.
                # Unless S is full?
                if current_size == n:
                    break
                # If we are stuck with no blocking APs, something is wrong or we are at a weird state.
                # Break to avoid infinite loop.
                break

            # Pick a random blocking AP
            b_idx = random.choice(blocking_aps)
            ap = ap_list[b_idx]
            
            # Find which 3 elements are in S
            in_ap = []
            for x in ap:
                if S_mask & (1 << (x - 1)):
                    in_ap.append(x)
                    
            if len(in_ap) != 3:
                # Should not happen if get_blocking_aps is correct
                continue
                
            # Pick a random element y from in_ap to remove
            y = random.choice(in_ap)
            
            # Tentatively remove y
            y_bit = 1 << (y - 1)
            S_mask_temp = S_mask & ~y_bit
            
            # Try to add any x not in S_mask_temp
            # We prioritize x that were blocked by this specific AP?
            # Actually, removing y unblocks AP b_idx.
            # Any x that was ONLY blocked by APs containing y might now be addable.
            # But simply trying all x is robust.
            
            added_x = None
            # To speed up, we can check x in random order or specific order.
            # Let's iterate all x not in S_mask_temp.
            for x in range(1, n + 1):
                if not (S_mask_temp & (1 << (x - 1))):
                    if can_add(x, S_mask_temp):
                        added_x = x
                        break
                        
            if added_x is not None:
                # Accept swap
                S_mask = S_mask_temp | (1 << (added_x - 1))
                # Size is unchanged, but configuration improved (hopefully)
                improved = True
                no_improve_counter = 0
            else:
                # Swap failed, revert (S_mask remains unchanged)
                # No improvement
                no_improve_counter += 1
                if no_improve_counter > max_no_improve:
                    break # Restart this local search phase
                    
        # End local search for this restart
        
    return sorted(best_set)
