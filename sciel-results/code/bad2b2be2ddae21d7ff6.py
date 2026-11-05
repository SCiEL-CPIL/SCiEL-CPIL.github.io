import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    n = 512
    
    # Precompute all 4-APs
    # ap_list: list of tuples (a, b, c, d)
    # conflicts: dict mapping element -> list of APs containing it
    # ap_masks: dict mapping element -> list of masks for APs containing it
    ap_list = []
    conflicts = {i: [] for i in range(1, n + 1)}
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
                conflicts[term].append(ap)
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

    best_set = set()
    best_size = 0
    best_mask = 0
    
    # Multi-start local search
    # We will run several restarts.
    # Each restart:
    # 1. Greedy fill (deterministic or random)
    # 2. Local search (additions and swaps)
    
    while time.time() < deadline:
        # 1. Greedy Construction
        # Using a random shuffle is good for diversity.
        # However, a "greedy by density" or simple ascending can be strong.
        # Let's stick to random shuffle for robustness but make it fast.
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
            
        # 2. Local Search (Swaps and Additions)
        # We alternate between trying to add (increase size) and swap (maintain size, change config).
        # Since we just did a greedy fill, there might be no immediate additions possible.
        # So we focus on swaps to reach a local optimum, then try additions again.
        
        steps_in_local_search = 0
        max_local_steps = 20000 # Increased due to speedup
        
        # Track if we made any progress recently to avoid getting stuck in a loop
        last_improvement_step = 0
        
        while steps_in_local_search < max_local_steps and time.time() < deadline:
            steps_in_local_search += 1
            
            # Check for additions first (cheap if we just failed a swap, but let's check periodically)
            # Actually, if we are in a local optimum for additions, we shouldn't check every step.
            # But checking is fast. Let's do it every 10 steps or if we just did a swap.
            
            # Try to add a random element
            if steps_in_local_search % 5 == 0:
                # Pick a few random candidates to try adding
                for _ in range(10):
                    x = random.randint(1, n)
                    if not (S_mask & (1 << (x - 1))):
                        if can_add(x, S_mask):
                            S_mask |= (1 << (x - 1))
                            S_size += 1
                            last_improvement_step = steps_in_local_search
                            if S_size > best_size:
                                best_size = S_size
                                best_mask = S_mask
                            break # Restart the inner loop or continue? Continue to try more adds.
            
            # Try a swap
            # Pick a random element x not in S
            x = random.randint(1, n)
            if S_mask & (1 << (x - 1)):
                continue
                
            # Check if x can be added directly (just in case)
            if can_add(x, S_mask):
                S_mask |= (1 << (x - 1))
                S_size += 1
                if S_size > best_size:
                    best_size = S_size
                    best_mask = S_mask
                continue
                
            # If x cannot be added, find a conflicting AP
            # We need to find an AP containing x where the other 3 are in S.
            conflicting_ap_mask = None
            for mask in ap_masks[x]:
                others_mask = mask ^ (1 << (x - 1))
                if (S_mask & others_mask) == others_mask:
                    conflicting_ap_mask = others_mask
                    break
                    
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
                # Using bit_length - 1 gives 0-indexed
                idx = lsb.bit_length() - 1
                bits_to_remove.append(idx + 1)
                temp_mask ^= lsb
                
            y = random.choice(bits_to_remove)
            bit_y = 1 << (y - 1)
            
            # Try swap: remove y, add x
            new_S_mask = (S_mask & ~bit_y) | (1 << (x - 1))
            
            # Check if new_S_mask is valid with respect to x
            # We know removing y breaks the specific conflicting AP.
            # But x might form a NEW AP with the rest of S_mask \ {y}.
            # So we need to check can_add(x, S_mask & ~bit_y).
            
            if can_add(x, S_mask & ~bit_y):
                S_mask = new_S_mask
                # Size is unchanged.
                # This new configuration might allow future additions.
            else:
                # Swap failed. Do nothing.
                pass

    # Convert best_mask to sorted list
    best_set = [i + 1 for i in range(n) if (best_mask >> i) & 1]
    return sorted(best_set)
