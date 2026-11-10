import time
import random

N = 256
# EVOLVE-BLOCK-START

def _construct_greedy_bitmask(rng, N):
    """
    Greedy construction using bitmasks.
    Selects elements that block the most other available elements.
    """
    S_mask = 0
    sums_mask = 0
    # Bits 1..N are available initially
    available_mask = ((1 << (N + 1)) - 1) ^ 1  # Clear bit 0
    
    S_list = []
    
    while available_mask:
        # Get list of available candidates
        avail_list = []
        temp_mask = available_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            x = lsb.bit_length() - 1
            avail_list.append(x)
            temp_mask ^= lsb
        
        if not avail_list:
            break
            
        # Shuffle to break ties and introduce randomness
        random.shuffle(avail_list)
        
        best_x = -1
        best_score = -1
        best_noise = 0.0
        
        # Precompute S_list for faster iteration in scoring
        S_current = []
        temp_S = S_mask
        while temp_S:
            lsb_S = temp_S & -temp_S
            a = lsb_S.bit_length() - 1
            S_current.append(a)
            temp_S ^= lsb_S
            
        for x in avail_list:
            # Check validity of x
            # 1. 2x not in sums
            if (sums_mask >> (2 * x)) & 1:
                continue
            # 2. No collision x+a with existing sums
            # (sums_mask >> x) gives bits for sums-x. If any of those correspond to an element in S, collision.
            if (sums_mask >> x) & S_mask:
                continue
                
            # Simulate adding x
            new_S_mask = S_mask | (1 << x)
            new_sums_mask = sums_mask | (1 << (2 * x))
            
            # Add x + a for all a in S
            # We can do this by iterating S_current or using bit tricks.
            # Since S is small, iterating is fine.
            for a in S_current:
                new_sums_mask |= (1 << (x + a))
                
            # Calculate blocking score: how many other available y become blocked?
            # A y is blocked if:
            # 1. 2y is in new_sums_mask
            # 2. There exists a' in new_S_mask such that y + a' is in new_sums_mask
            #    This is equivalent to: (new_sums_mask >> y) & new_S_mask != 0
            
            blocked_count = 0
            for y in avail_list:
                if y == x:
                    continue
                
                # Check if y is blocked by new state
                # Condition 1: 2y in new_sums
                if (new_sums_mask >> (2 * y)) & 1:
                    blocked_count += 1
                    continue
                    
                # Condition 2: y + a' in new_sums for some a' in new_S
                if (new_sums_mask >> y) & new_S_mask:
                    blocked_count += 1
                    
            # Add small random noise to encourage diversity
            noise = rng.random() * 0.5
            score = blocked_count + noise
            
            if score > best_score:
                best_score = score
                best_x = x
                best_noise = noise
        
        if best_x == -1:
            # Should not happen if available is not empty and we start with empty S,
            # but if no valid x found, we are stuck.
            break
            
        # Commit best_x
        S_mask |= (1 << best_x)
        S_list.append(best_x)
        
        # Update sums_mask
        sums_mask |= (1 << (2 * best_x))
        # Add best_x + a for a in old S (S_current)
        for a in S_current:
            sums_mask |= (1 << (best_x + a))
            
        # Update available_mask
        # Remove best_x
        available_mask &= ~(1 << best_x)
        
        # Remove any y that is now blocked
        # Iterate remaining available
        temp_avail = available_mask
        new_avail_mask = 0
        while temp_avail:
            lsb_y = temp_avail & -temp_avail
            y = lsb_y.bit_length() - 1
            # Check if y is still valid for current S
            # 1. 2y not in sums
            if not ((sums_mask >> (2 * y)) & 1):
                # 2. No collision y+a
                if not ((sums_mask >> y) & S_mask):
                    new_avail_mask |= lsb_y
            temp_avail ^= lsb_y
            
        available_mask = new_avail_mask
        
    return S_list

def _is_maximal_sidon_fast_bitmask(S, N):
    """
    Check if S is a maximal Sidon set in [1, N].
    S is assumed to be a valid Sidon set.
    """
    S_mask = 0
    sums_mask = 0
    
    # Build masks
    for a in S:
        S_mask |= (1 << a)
    
    n = len(S)
    for i in range(n):
        a = S[i]
        sums_mask |= (1 << (2 * a))
        for j in range(i + 1, n):
            b = S[j]
            sums_mask |= (1 << (a + b))
            
    # Check all missing elements
    for x in range(1, N + 1):
        if (S_mask >> x) & 1:
            continue
            
        # Check if x is blocked
        # 1. 2x in sums
        if (sums_mask >> (2 * x)) & 1:
            continue
            
        # 2. x + a in sums for some a in S
        # (sums_mask >> x) gives bits for sums-x. Check intersection with S_mask.
        if (sums_mask >> x) & S_mask:
            continue
            
        # If we reach here, x is not blocked, so S is not maximal
        return False
        
    return True

def _shrink_maximal_bitmask(S, N, deadline):
    """
    Try to remove elements from S while maintaining maximality.
    Returns the shrunk list.
    """
    S = list(S)
    
    # Precompute initial sums mask for faster checks? 
    # Actually, recomputing is cheap for small S.
    
    while time.time() < deadline:
        improved = False
        indices = list(range(len(S)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
                
            # Tentative removal
            candidate = S[:idx] + S[idx+1:]
            
            # Check if candidate is maximal
            if _is_maximal_sidon_fast_bitmask(candidate, N):
                S = candidate
                improved = True
                break
        
        if not improved:
            break
            
    return S

def construct():
    start_time = time.time()
    # Safety margin: 55s out of 60s limit
    deadline = start_time + 55.0
    
    best = None
    best_size = float('inf')
    
    # We will run multiple restarts
    # Each restart uses a different seed
    # The number of restarts is not fixed, but limited by time
    iter_count = 0
    
    while time.time() < deadline:
        rng = random.Random(iter_count * 7919 + 104729)
        
        # 1. Greedy construction
        sel = _construct_greedy_bitmask(rng, N)
        
        if not sel:
            iter_count += 1
            continue
            
        # 2. Shrinkage
        # Allocate some time for shrinking, but ensure we have time for next restart
        # If we are close to deadline, shrink less aggressively or skip if no time
        shrink_deadline = min(deadline - 1.0, time.time() + 5.0) # Cap shrink time per restart at 5s
        
        sel = _shrink_maximal_bitmask(sel, N, shrink_deadline)
        
        # 3. Update best
        if len(sel) < best_size:
            best = sel
            best_size = len(sel)
            
            # If we find a very small set, we might stop early?
            # Theoretical lower bound for Sidon set in [1,N] is ~sqrt(N).
            # For N=256, sqrt(256)=16.
            # Maximal Sidon sets can be smaller? No, maximal means cannot add.
            # A Sidon set of size k has ~k^2/2 sums.
            # To block all N elements, we need enough sums.
            # If best_size <= 16, it's likely near optimal for this N.
            if best_size <= 16:
                break
                
        iter_count += 1
        
    if best is None:
        # Fallback: just return a valid maximal set (greedy without shrink)
        sel = _construct_greedy_bitmask(random.Random(0), N)
        return sel
        
    return best

# EVOLVE-BLOCK-END
