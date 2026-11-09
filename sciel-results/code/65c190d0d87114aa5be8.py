import time
import random

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0  # Safety margin

    # Precompute conflicts as bitmasks
    # conflicts[x] is a list of bitmasks. 
    # Each bitmask represents 3 indices (i, j, k).
    # If S contains all 3 bits, x cannot be added.
    
    # Generate all 4-APs
    # 0-indexed: a, a+d, a+2d, a+3d
    conflicts = [[] for _ in range(N)]
    
    for d in range(1, (N // 3) + 1):
        for a in range(1, N - 3 * d + 1):
            # 1-indexed a -> 0-indexed a-1
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            
            # For i0, others are i1, i2, i3
            mask0 = (1 << i1) | (1 << i2) | (1 << i3)
            conflicts[i0].append(mask0)
            
            # For i1, others are i0, i2, i3
            mask1 = (1 << i0) | (1 << i2) | (1 << i3)
            conflicts[i1].append(mask1)
            
            # For i2, others are i0, i1, i3
            mask2 = (1 << i0) | (1 << i1) | (1 << i3)
            conflicts[i2].append(mask2)
            
            # For i3, others are i0, i1, i2
            mask3 = (1 << i0) | (1 << i1) | (1 << i2)
            conflicts[i3].append(mask3)

    def can_add(x, S_mask):
        """
        Check if x (0-indexed) can be added to S (represented as integer bitmask).
        """
        if (S_mask >> x) & 1:
            return False
        for cm in conflicts[x]:
            if (S_mask & cm) == cm:
                return False
        return True

    def get_blocking_mask(x, S_mask):
        """
        Return a random blocking mask for x, or None if x can be added.
        """
        if (S_mask >> x) & 1:
            return None
        for cm in conflicts[x]:
            if (S_mask & cm) == cm:
                return cm
        return None

    def greedy_construct(seed):
        rng = random.Random(seed)
        S_mask = 0
        elements = list(range(N))
        rng.shuffle(elements)
        
        for x in elements:
            if can_add(x, S_mask):
                S_mask |= (1 << x)
        return S_mask

    def local_search(S_mask, rng, time_budget):
        S_mask = S_mask
        current_size = bin(S_mask).count('1')
        best_size = current_size
        best_mask = S_mask
        
        while time.time() - start_time < time_budget:
            x = rng.randint(0, N - 1)
            
            if (S_mask >> x) & 1:
                continue # Already in set
                
            blocking = get_blocking_mask(x, S_mask)
            
            if blocking is None:
                # Can add
                S_mask |= (1 << x)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_mask = S_mask
            else:
                # Blocked by 'blocking' mask
                # Pick a random bit from 'blocking' to remove
                # 'blocking' has 3 bits set.
                # Extract the 3 indices
                bits = []
                b = blocking
                while b:
                    lsb = b & -b
                    idx = lsb.bit_length() - 1
                    bits.append(idx)
                    b -= lsb
                # bits has 3 elements
                y = bits[rng.randint(0, 2)]
                
                # Remove y
                S_mask &= ~(1 << y)
                current_size -= 1
                
                # Try to add x
                if can_add(x, S_mask):
                    S_mask |= (1 << x)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = S_mask
                else:
                    # Still blocked, or maybe we removed y but x is blocked by another triple
                    # If current_size < best_size - 5, revert?
                    # We don't track "revert" easily with just mask, but we can just continue.
                    # To prevent shrinking, we can check if size dropped too much.
                    if current_size < best_size - 10:
                        S_mask = best_mask
                        current_size = best_size

        return best_mask

    best_mask = 0
    best_size = 0
    rng = random.Random(42)
    
    # Multiple restarts
    # Bitmask operations are very fast, so we can do many restarts.
    for seed in range(50):
        if time.time() - start_time > time_limit - 5:
            break
            
        S_mask = greedy_construct(seed)
        size = bin(S_mask).count('1')
        
        if size > best_size:
            best_size = size
            best_mask = S_mask
            
        # Local search
        # Allocate time based on remaining
        remaining = time_limit - (time.time() - start_time)
        # Give each local search a small chunk, e.g., 0.5s
        # Or just let it run until the main loop breaks?
        # The local search has its own time check.
        S_mask = local_search(S_mask, rng, time_limit - 2)
        size = bin(S_mask).count('1')
        
        if size > best_size:
            best_size = size
            best_mask = S_mask

    # Final polish: try to add any remaining elements in order
    for x in range(N):
        if not ((best_mask >> x) & 1) and can_add(x, best_mask):
            best_mask |= (1 << x)
            best_size += 1

    # Convert to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m -= lsb
        
    return result
