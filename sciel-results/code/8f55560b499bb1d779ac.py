import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    n = 512
    
    # Precompute all 4-APs
    # ap_list: list of tuples (a, b, c, d)
    # ap_masks: dict mapping element -> list of masks for APs containing it
    # The mask for an AP is a 512-bit integer where bits corresponding to the AP terms are set.
    ap_list = []
    ap_masks = {i: [] for i in range(1, n + 1)}
    
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            c = a + 2 * d
            b = a + d
            term4 = a + 3 * d
            if term4 > n:
                break
            ap = (a, b, c, term4)
            ap_list.append(ap)
            # Create bitmask for this AP
            mask = 0
            for term in ap:
                mask |= (1 << (term - 1))
                ap_masks[term].append(mask)
                
    def can_add(x, S_mask):
        """Check if x can be added to S_mask without forming a 4-AP."""
        bit_x = 1 << (x - 1)
        for mask in ap_masks[x]:
            # The other 3 elements are in mask ^ bit_x
            others_mask = mask ^ bit_x
            # If all 'others' are in S_mask, we have a conflict
            if (S_mask & others_mask) == others_mask:
                return False
        return True

    def get_conflicting_mask(x, S_mask):
        """Find the mask of the other 3 elements of a conflicting AP for x."""
        bit_x = 1 << (x - 1)
        for mask in ap_masks[x]:
            others_mask = mask ^ bit_x
            if (S_mask & others_mask) == others_mask:
                return others_mask
        return None

    def mask_to_list(mask):
        """Convert a bitmask to a list of 1-based indices."""
        return [i + 1 for i in range(n) if (mask >> i) & 1]

    best_size = 0
    best_mask = 0
    
    # Strategy: Multi-start local search.
    # We use the base-3 Cantor set as a strong initial seed for the first few iterations,
    # and random greedy for subsequent iterations to explore different basins.
    
    # Precompute the Cantor set mask (numbers with no '1' in base 3)
    cantor_mask = 0
    for i in range(1, n + 1):
        x = i
        valid = True
        while x > 0:
            if x % 3 == 1:
                valid = False
                break
            x //= 3
        if valid:
            cantor_mask |= (1 << (i - 1))
            
    # We will run several restarts.
    # The first restart uses the Cantor set. Subsequent ones use random greedy.
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        
        # 1. Initialize S
        if restarts == 1:
            S_mask = cantor_mask
        else:
            # Random greedy initialization
            S_mask = 0
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
            for x in candidates:
                if can_add(x, S_mask):
                    S_mask |= (1 << (x - 1))
                    
        S_size = bin(S_mask).count('1')
        if S_size > best_size:
            best_size = S_size
            best_mask = S_mask
            
        # 2. Local Search
        # We alternate between trying to add elements and performing swaps.
        # Since the initial set (especially Cantor) might not be maximal, 
        # we first aggressively try to add elements.
        
        steps_in_local_search = 0
        max_local_steps = 50000 # Increased limit due to speed of bitmask ops
        
        # Track if we made any progress recently to detect local optima for additions
        last_add_step = 0
        no_add_count = 0
        max_no_add = 500 # If we fail to add for this many steps, focus on swaps
        
        while steps_in_local_search < max_local_steps and time.time() < deadline:
            steps_in_local_search += 1
            
            # Determine mode: Add or Swap
            # If we haven't tried adding recently or are in 'add' phase, try adding.
            # If we've failed to add many times, switch to 'swap' mode.
            
            if no_add_count < max_no_add:
                # Try to add a random element
                # To speed up, we can pick from a precomputed list of non-selected elements
                # but random.randint is simple and fast enough for N=512.
                x = random.randint(1, n)
                if not (S_mask & (1 << (x - 1))):
                    if can_add(x, S_mask):
                        S_mask |= (1 << (x - 1))
                        S_size += 1
                        last_add_step = steps_in_local_search
                        no_add_count = 0
                        if S_size > best_size:
                            best_size = S_size
                            best_mask = S_mask
                        continue
                    else:
                        no_add_count += 1
                # If x was already in S, or we couldn't add it, we fall through to swap logic
                # or just continue if we are still in add phase.
                # Actually, if we are in add phase and fail, we should probably try a swap
                # to escape the local optimum.
                
            # Try a swap
            # Pick a random element x not in S
            x = random.randint(1, n)
            if S_mask & (1 << (x - 1)):
                continue
                
            # Check if x can be added directly (just in case, though we just checked above)
            if can_add(x, S_mask):
                S_mask |= (1 << (x - 1))
                S_size += 1
                last_add_step = steps_in_local_search
                no_add_count = 0
                if S_size > best_size:
                    best_size = S_size
                    best_mask = S_mask
                continue
                
            # If x cannot be added, find a conflicting AP
            conflicting_ap_mask = get_conflicting_mask(x, S_mask)
            
            if not conflicting_ap_mask:
                # Should not happen if can_add returned False
                continue
                
            # Pick a random element y from the conflicting AP to remove
            # The conflicting AP consists of x and the 3 elements in conflicting_ap_mask.
            # We need to pick one of the 3 elements in conflicting_ap_mask.
            # Get the bits set in conflicting_ap_mask.
            bits_to_remove = []
            temp_mask = conflicting_ap_mask
            while temp_mask:
                # Get lowest set bit
                lsb = temp_mask & -temp_mask
                # Index is log2(lsb) + 1
                idx = lsb.bit_length() - 1
                bits_to_remove.append(idx + 1)
                temp_mask ^= lsb
                
            y = random.choice(bits_to_remove)
            bit_y = 1 << (y - 1)
            
            # Try swap: remove y, add x
            # We need to check if x can be added to S \ {y}
            new_S_mask = (S_mask & ~bit_y) | (1 << (x - 1))
            
            if can_add(x, S_mask & ~bit_y):
                S_mask = new_S_mask
                # Size is unchanged.
                # Reset no_add_count because the configuration changed, 
                # potentially opening up new addition opportunities.
                no_add_count = max(0, no_add_count // 2)
            else:
                # Swap failed. Do nothing.
                pass

    # Convert best_mask to sorted list
    best_set = [i + 1 for i in range(n) if (best_mask >> i) & 1]
    return sorted(best_set)
