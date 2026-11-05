import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety

    N = 512
    K = 32
    MAX_D = 255

    def get_score(m):
        """
        Calculate the number of distinct d > 0 such that 
        a, a+d, a+2d are in the set represented by bitmask m.
        """
        score = 0
        # d can be at most 255 because 2*255 = 510 < 512
        # If d=256, 2d=512, which is out of bounds for 0..511 indices (values 1..512)
        for d in range(1, MAX_D + 1):
            if (m & (m >> d) & (m >> (2 * d))) != 0:
                score += 1
        return score

    def get_set_bits(m):
        return [i for i in range(N) if (m >> i) & 1]

    def get_unset_bits(m, set_bits):
        # Optimization: We know the size, but generating a set is faster for lookup?
        # Actually, for the swap loop, we need a list of unset bits.
        # Let's just generate it. 480 items.
        s_set = set(set_bits)
        return [i for i in range(N) if i not in s_set]

    def init_block(offset=0):
        """Initialize with a contiguous block."""
        mask = 0
        for i in range(K):
            mask |= (1 << (offset + i))
        return mask

    def init_sparse():
        """Initialize with a sparse arithmetic progression."""
        mask = 0
        step = N // K
        for i in range(K):
            mask |= (1 << (i * step))
        return mask

    def init_random():
        """Initialize with random elements."""
        indices = random.sample(range(N), K)
        mask = 0
        for i in indices:
            mask |= (1 << i)
        return mask

    best_mask = 0
    best_score = -1

    # Define initializers to try
    # We will try a few different starting points in each restart
    initializers = [
        lambda: init_block(0),
        lambda: init_block(240), # Middle
        lambda: init_sparse(),
        lambda: init_random()
    ]
    
    # Number of restarts. 
    # Each restart consists of SA + Polish.
    # Polish is the heavy hitter. 
    # Let's aim for 3-5 restarts.
    num_restarts = 5
    # time_per_restart is calculated dynamically based on remaining time?
    # Or fixed fraction? 
    # Let's use a fixed fraction but check deadline inside.
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Pick an initializer
        init_func = initializers[restart % len(initializers)]
        mask = init_func()
        
        current_score = get_score(mask)
        restart_best_mask = mask
        restart_best_score = current_score
        
        # --- Phase 1: Simulated Annealing ---
        # Parameters
        T = 10.0
        T_min = 0.1
        cooling_rate = 0.999
        max_sa_steps = 20000 # Cap steps to prevent infinite loop if time check fails?
        # Actually, time check is inside.
        
        set_bits = get_set_bits(mask)
        unset_bits = get_unset_bits(mask, set_bits)
        
        sa_start = time.time()
        sa_deadline = sa_start + (deadline - sa_start) * 0.7 # Use 70% of remaining time for SA
        
        steps = 0
        while T > T_min and time.time() < sa_deadline:
            # Pick random i from set_bits, j from unset_bits
            idx_i = random.randint(0, len(set_bits) - 1)
            idx_j = random.randint(0, len(unset_bits) - 1)
            
            i = set_bits[idx_i]
            j = unset_bits[idx_j]
            
            # Swap
            new_mask = mask ^ (1 << i) ^ (1 << j)
            new_score = get_score(new_mask)
            
            delta = new_score - current_score
            
            if delta >= 0:
                # Accept
                mask = new_mask
                current_score = new_score
                set_bits[idx_i] = j
                unset_bits[idx_j] = i
                
                if current_score > restart_best_score:
                    restart_best_score = current_score
                    restart_best_mask = mask
                    
            else:
                # Accept with probability exp(delta / T)
                if random.random() < (2.718281828 ** (delta / T)):
                    mask = new_mask
                    current_score = new_score
                    set_bits[idx_i] = j
                    unset_bits[idx_j] = i
            
            T *= cooling_rate
            steps += 1
            
            # Update global best if this restart is beating it
            if restart_best_score > best_score:
                best_score = restart_best_score
                best_mask = restart_best_mask

        # --- Phase 2: Greedy Polish ---
        # Try to improve by finding the best swap.
        # We do this until no improvement is found or time runs out.
        
        # Re-initialize lists for polish
        set_bits = get_set_bits(mask)
        unset_bits = get_unset_bits(mask, set_bits)
        
        polish_deadline = deadline - 2.0 # Leave 2s for finalization
        
        while time.time() < polish_deadline:
            best_delta = 0
            best_i = -1
            best_j = -1
            best_new_score = current_score
            
            # Iterate over all possible swaps
            # To speed this up, we can break early if we find a big improvement?
            # No, "best improvement" requires checking all.
            # 32 * 480 = 15360 swaps.
            # Each swap costs ~1 microsecond for score? 
            # 15360 * 1us = 15ms. 
            # So one full pass is very fast.
            
            for i in set_bits:
                for j in unset_bits:
                    new_mask = mask ^ (1 << i) ^ (1 << j)
                    new_score = get_score(new_mask)
                    delta = new_score - current_score
                    if delta > best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
                        best_new_score = new_score
            
            if best_delta > 0:
                # Apply the best swap
                mask = mask ^ (1 << best_i) ^ (1 << best_j)
                current_score = best_new_score
                
                # Update lists
                set_bits.remove(best_i)
                set_bits.append(best_j)
                unset_bits.remove(best_j)
                unset_bits.append(best_i)
                
                if current_score > restart_best_score:
                    restart_best_score = current_score
                    restart_best_mask = mask
                    
                if restart_best_score > best_score:
                    best_score = restart_best_score
                    best_mask = restart_best_mask
            else:
                # No improvement found, break polish loop
                break

    # Convert best_mask to list of integers
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    # Ensure exactly 32 distinct integers
    # The logic above guarantees this, but a safety check is good
    if len(result) != K:
        # Fallback to a known valid set (e.g., 1..32)
        result = list(range(1, K + 1))
        
    return result
