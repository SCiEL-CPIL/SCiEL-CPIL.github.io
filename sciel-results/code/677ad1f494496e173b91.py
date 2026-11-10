import time
import random

N = 256
MAX_SUM = 2 * N

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # Safety margin for 30s limit (or 60s if that's the real one, contract says 30s)
    
    # We want to minimize the size of the maximal Sidon set.
    # Score is seed/value, so we want value (size) to be small.
    # Seed is likely 1.0 or similar? "1.0 equals the seed program".
    # If seed program size is S_seed, score = S_seed / S_current.
    # We want S_current to be as small as possible.
    
    best_size = N + 1
    best_set = []
    
    # Bitmask helpers
    # We use bits 1..N for elements. Bit 0 is unused.
    # Sums use bits 2..2N.
    
    def get_avail_list(mask):
        """Extract list of indices from bitmask."""
        lst = []
        m = mask
        while m:
            lsb = m & -m
            lst.append(lsb.bit_length() - 1)
            m ^= lsb
        return lst

    def is_sidon_mask(S_mask, sums_mask):
        """
        Verify if S_mask represents a Sidon set with given sums_mask.
        This is a sanity check, not used in the fast loop.
        """
        S_list = get_avail_list(S_mask)
        S_set = set(S_list)
        sums_set = set()
        for i, a in enumerate(S_list):
            for j in range(i, len(S_list)):
                s = a + S_list[j]
                if s in sums_set:
                    return False
                sums_set.add(s)
        return sums_set == set(get_avail_list(sums_mask))

    def greedy_extend(S_mask, sums_mask, available_mask, rng):
        """
        Greedily extend S to a maximal Sidon set.
        Returns (new_S_mask, new_sums_mask, new_available_mask, S_list).
        """
        S_list = get_avail_list(S_mask)
        
        while available_mask:
            avail_list = get_avail_list(available_mask)
            if not avail_list:
                break
            
            # Shuffle for randomness
            rng.shuffle(avail_list)
            
            best_z = -1
            max_blocked = -1
            best_score = -1.0
            
            for z in avail_list:
                # Check if z is valid to add to current S
                # 1. 2z not in sums
                if (sums_mask >> (2*z)) & 1:
                    continue
                # 2. No collision: (sums >> z) & S_mask == 0
                if (sums_mask >> z) & S_mask:
                    continue
                
                # Simulate adding z
                new_S_mask = S_mask | (1 << z)
                new_sums_mask = sums_mask | (1 << (2*z))
                
                # Add z+a for a in S
                # Iterate through S
                temp_S = S_mask
                add_bits = 0
                while temp_S:
                    lsb_S = temp_S & -temp_S
                    a = lsb_S.bit_length() - 1
                    add_bits |= (1 << (z + a))
                    temp_S ^= lsb_S
                new_sums_mask |= add_bits
                
                # Count how many other available elements are blocked
                # An element w is blocked if:
                # 1. 2w in new_sums_mask
                # 2. (new_sums_mask >> w) & new_S_mask != 0
                blocked_count = 0
                # We only care about w in available_mask, w != z
                # To speed up, we can iterate avail_list
                for w in avail_list:
                    if w == z:
                        continue
                    # Check if w is still available (it is, by definition of loop start, 
                    # but we are checking if it becomes BLOCKED)
                    # If it is blocked, it will be removed from available in the next step.
                    
                    # Check block condition 1
                    if (new_sums_mask >> (2*w)) & 1:
                        blocked_count += 1
                        continue
                    # Check block condition 2
                    if (new_sums_mask >> w) & new_S_mask:
                        blocked_count += 1
                
                # Heuristic: Maximize blocked count.
                # Add noise to encourage diversity
                score = blocked_count + rng.random() * 0.5
                
                if score > best_score:
                    best_score = score
                    best_z = z
            
            if best_z == -1:
                # No valid z found? Should not happen if available is non-empty and S is Sidon.
                # But maybe all available are invalid?
                break
            
            # Commit best_z
            S_mask = S_mask | (1 << best_z)
            S_list.append(best_z)
            
            # Update sums_mask
            sums_mask |= (1 << (2*best_z))
            temp_S = S_mask ^ (1 << best_z) # Old S
            while temp_S:
                lsb_S = temp_S & -temp_S
                a = lsb_S.bit_length() - 1
                sums_mask |= (1 << (best_z + a))
                temp_S ^= lsb_S
            
            # Update available_mask
            available_mask &= ~(1 << best_z)
            
            # Remove blocked elements
            new_avail_mask = 0
            temp_avail = available_mask
            while temp_avail:
                lsb_w = temp_avail & -temp_avail
                w = lsb_w.bit_length() - 1
                
                # Check if w is still valid
                # 1. 2w not in sums_mask
                if not ((sums_mask >> (2*w)) & 1):
                    # 2. No collision
                    if not ((sums_mask >> w) & S_mask):
                        new_avail_mask |= lsb_w
                temp_avail ^= lsb_w
            
            available_mask = new_avail_mask
            
        return S_mask, sums_mask, available_mask, S_list

    # Main Loop
    for iter in range(500):
        if time.time() > deadline:
            break
            
        rng = random.Random(iter)
        
        # Start with empty set
        S_mask = 0
        sums_mask = 0
        available_mask = ((1 << (N + 1)) - 1) ^ 1  # Bits 1..N
        
        # Initial Greedy Extension
        S_mask, sums_mask, available_mask, S_list = greedy_extend(S_mask, sums_mask, available_mask, rng)
        
        current_size = len(S_list)
        
        # Local Search: Try to remove elements and re-extend
        # We try to remove an element x from S, then greedily extend the remainder.
        # If the new maximal set is smaller, we accept it.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list_copy = S_list[:]
            rng.shuffle(S_list_copy)
            
            for x in S_list_copy:
                if time.time() > deadline:
                    break
                
                # Remove x from S
                new_S_mask = S_mask ^ (1 << x)
                
                # Recalculate sums for new_S_mask
                # This is O(|S|^2) which is small
                new_sums_mask = 0
                new_S_list = [s for s in S_list if s != x]
                for i, a in enumerate(new_S_list):
                    for j in range(i, len(new_S_list)):
                        new_sums_mask |= (1 << (a + new_S_list[j]))
                
                # Recalculate available mask for new_S_mask
                new_avail_mask = 0
                for w in range(1, N + 1):
                    if (new_S_mask >> w) & 1:
                        continue # w is in S
                    # Check if w is valid
                    if not ((new_sums_mask >> (2*w)) & 1):
                        if not ((new_sums_mask >> w) & new_S_mask):
                            new_avail_mask |= (1 << w)
                
                # Greedily extend new_S_mask
                new_S_mask_ext, new_sums_mask_ext, _, new_S_list_ext = greedy_extend(
                    new_S_mask, new_sums_mask, new_avail_mask, rng
                )
                
                if len(new_S_list_ext) < current_size:
                    # Accept improvement
                    S_mask = new_S_mask_ext
                    sums_mask = new_sums_mask_ext
                    S_list = new_S_list_ext
                    current_size = len(S_list)
                    improved = True
                    break # Restart the while loop with new S
        
        if current_size < best_size:
            best_size = current_size
            best_set = S_list[:]
            
    return best_set
