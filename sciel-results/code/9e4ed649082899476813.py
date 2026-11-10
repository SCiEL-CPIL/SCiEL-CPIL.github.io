import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25.0
    
    best_size = N + 1
    best_set = []
    
    # We will run multiple random restarts
    # The greedy choice is: pick z that blocks the most available elements.
    # To introduce randomness, we can shuffle available or add noise to the score.
    
    for iter in range(500): # Increase iterations due to speedup
        if time.time() > deadline:
            break
            
        random.seed(iter)
        
        S_mask = 0
        sums_mask = 0
        # Bits 1..N set.
        available_mask = ((1 << (N + 1)) - 1) ^ 1
        
        S_list = []
        
        while available_mask:
            # Find best z
            best_z = -1
            max_blocked = -1
            
            # Get list of available elements
            avail_list = []
            temp_mask = available_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                z = lsb.bit_length() - 1
                avail_list.append(z)
                temp_mask ^= lsb
            
            random.shuffle(avail_list)
            
            # Precompute list of elements in S for faster iteration
            S_elements = []
            temp_S = S_mask
            while temp_S:
                lsb_S = temp_S & -temp_S
                a = lsb_S.bit_length() - 1
                S_elements.append(a)
                temp_S ^= lsb_S
            
            for z in avail_list:
                # Check if z is valid to add
                # 1. 2z not in sums
                if (sums_mask >> (2*z)) & 1:
                    continue
                # 2. No collision with existing sums
                if (sums_mask >> z) & S_mask:
                    continue
                
                # Count newly blocked w in available_mask
                # A w is blocked if:
                # 1. 2w == z+a for some a in S
                # 2. w+a == 2z for some a in S
                # 3. w+a == z+b for some a,b in S
                
                blocked_mask = 0
                # Condition 1: 2w = z+a => w = (z+a)/2
                for a in S_elements:
                    s = z + a
                    if s % 2 == 0:
                        w = s // 2
                        if 1 <= w <= N:
                            blocked_mask |= (1 << w)
                    # Condition 2: w+a = 2z => w = 2z-a
                    w2 = 2*z - a
                    if 1 <= w2 <= N:
                        blocked_mask |= (1 << w2)
                
                # Condition 3: w+a = z+b => w = z+b-a
                for a in S_elements:
                    for b in S_elements:
                        w3 = z + b - a
                        if 1 <= w3 <= N:
                            blocked_mask |= (1 << w3)
                
                # Intersect with available_mask to count only those that were actually available
                # Note: z itself is in available_mask, but w=z is excluded by logic above?
                # If w=z, it means z is blocked by adding z?
                # 2z = z+a => z=a (no, a in S, z avail)
                # z+a = 2z => a=z (no)
                # z+a = z+b => a=b (yes, if a=b, w=z).
                # So z will be in blocked_mask if there exists a,b in S such that z+b-a = z => b=a.
                # Since a,b iterate all pairs, b=a is included. So z is always in blocked_mask.
                # We should exclude z from the count because we are adding z, not blocking it.
                # Or just count bits in (blocked_mask & available_mask) and subtract 1 if z is set.
                
                potential_blocked = blocked_mask & available_mask
                count = potential_blocked.bit_count()
                if (potential_blocked >> z) & 1:
                    count -= 1 # Exclude z itself
                
                # Heuristic: Maximize blocked_count
                # Add small random noise to break ties and encourage diversity
                score = count + random.random() * 0.5
                
                if score > max_blocked:
                    max_blocked = score
                    best_z = z
            
            if best_z == -1:
                break
            
            # Commit best_z
            S_mask |= (1 << best_z)
            S_list.append(best_z)
            
            # Update sums_mask
            # Add 2*best_z
            sums_mask |= (1 << (2*best_z))
            # Add best_z + a for a in old S
            temp_S = S_mask ^ (1 << best_z) # Old S
            while temp_S:
                lsb_S = temp_S & -temp_S
                a = lsb_S.bit_length() - 1
                sums_mask |= (1 << (best_z + a))
                temp_S ^= lsb_S
            
            # Update available_mask
            # Remove best_z
            available_mask &= ~(1 << best_z)
            
            # Remove any w that is now blocked
            # We can use the same logic to compute the new blocked set for the *actual* best_z
            # Or just re-check all available w's.
            # Given the speedup, re-checking might be fine, but let's use the fast method for consistency?
            # Actually, the fast method computed `blocked_mask` for `z`.
            # We can reuse the logic.
            
            # Re-calculate blocked mask for best_z to update available_mask efficiently?
            # Or just iterate available_mask and check validity.
            # The previous code iterated all avail. Let's stick to that for correctness simplicity,
            # but it's O(M). M is decreasing.
            
            temp_avail = available_mask
            new_avail_mask = 0
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
        
        if len(S_list) < best_size:
            best_size = len(S_list)
            best_set = S_list[:]
            
    return best_set
