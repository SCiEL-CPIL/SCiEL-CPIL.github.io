import time
import random

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0  # Safe margin before 90s kill

    # Precompute conflict masks using Python integers
    # For each element x (0-indexed), conflict_masks[x] is a list of masks.
    # Each mask represents the other 3 elements in a 4-AP containing x.
    # x can be added iff for NO mask m in conflict_masks[x]: (S_mask & m) == m.
    
    conflict_masks = [[] for _ in range(N)]
    
    for d in range(1, (N // 3) + 1):
        for a in range(0, N - 3 * d):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            
            m0 = (1 << ap[1]) | (1 << ap[2]) | (1 << ap[3])
            m1 = (1 << ap[0]) | (1 << ap[2]) | (1 << ap[3])
            m2 = (1 << ap[0]) | (1 << ap[1]) | (1 << ap[3])
            m3 = (1 << ap[0]) | (1 << ap[1]) | (1 << ap[2])
            
            conflict_masks[ap[0]].append(m0)
            conflict_masks[ap[1]].append(m1)
            conflict_masks[ap[2]].append(m2)
            conflict_masks[ap[3]].append(m3)
    
    # Convert to tuples for slightly faster iteration
    conflict_masks = [tuple(m) for m in conflict_masks]
    
    def can_add(x, S_mask):
        """Check if x (0-indexed) can be added to set represented by S_mask (int bitmask)"""
        if (S_mask >> x) & 1:
            return False
        for m in conflict_masks[x]:
            if (S_mask & m) == m:
                return False
        return True
    
    def get_size(S_mask):
        return bin(S_mask).count('1')
    
    def greedy_construct(order):
        S_mask = 0
        for x in order:
            if can_add(x, S_mask):
                S_mask |= (1 << x)
        return S_mask
    
    def local_search(S_mask, rng, time_budget):
        current_size = get_size(S_mask)
        best_size = current_size
        best_S_mask = S_mask
        
        all_elements = list(range(N))
        
        while time.time() - start_time < time_budget:
            # Randomly pick an element to remove
            if current_size == 0:
                break
            
            # Pick a random element from the current set to remove
            # Using random index into a shuffled list of set elements
            # For efficiency, we just pick a random x in [0, N) and check if it's in S
            # If not, try again (up to a few times)
            x_remove = -1
            for _ in range(10):
                x_remove = rng.randint(0, N - 1)
                if (S_mask >> x_remove) & 1:
                    break
            if x_remove == -1:
                continue
                
            S_mask &= ~(1 << x_remove)
            current_size -= 1
            
            # Try to add any element
            # Shuffle candidates for diversity
            candidates = all_elements[:]
            rng.shuffle(candidates)
            
            for x in candidates:
                if can_add(x, S_mask):
                    S_mask |= (1 << x)
                    current_size += 1
            
            if current_size > best_size:
                best_size = current_size
                best_S_mask = S_mask
        
        return best_S_mask

    # Main
    best_S_mask = 0
    best_size = 0
    
    rng = random.Random(42)
    
    # Generate diverse initial orders
    orders = []
    
    # Natural order
    orders.append(list(range(N)))
    
    # Reverse order
    orders.append(list(range(N - 1, -1, -1)))
    
    # Bit-reversal order
    bit_rev = []
    for i in range(N):
        # Reverse bits of i (considering up to 9 bits since 512 = 2^9)
        rev = 0
        for b in range(9):
            if (i >> b) & 1:
                rev |= (1 << (8 - b))
        bit_rev.append(rev)
    orders.append(bit_rev)
    
    # Random shuffles
    for _ in range(5):
        r = list(range(N))
        rng.shuffle(r)
        orders.append(r)
    
    # Modular patterns
    # e.g., i * 3 % 512, i * 5 % 512, etc. (must be coprime to 512 for permutation)
    # 512 = 2^9, so odd multipliers work.
    for mult in [3, 5, 7, 11, 13]:
        mod_order = [(i * mult) % N for i in range(N)]
        orders.append(mod_order)
    
    # Process each order with greedy + local search
    for idx, order in enumerate(orders):
        if time.time() - start_time > time_limit - 5:
            break
            
        S_mask = greedy_construct(order)
        size = get_size(S_mask)
        
        # Only do local search if we have time and the initial size is promising
        if time.time() - start_time < time_limit - 10 and size >= best_size - 5:
            S_mask = local_search(S_mask, rng, time_limit - 10)
            size = get_size(S_mask)
            
        if size > best_size:
            best_size = size
            best_S_mask = S_mask
    
    # If we have more time, do more random restarts
    seed = 100
    while time.time() - start_time < time_limit - 2:
        r = list(range(N))
        rng.shuffle(r)
        S_mask = greedy_construct(r)
        size = get_size(S_mask)
        
        if size >= best_size - 5 and time.time() - start_time < time_limit - 5:
            S_mask = local_search(S_mask, rng, time_limit - 5)
            size = get_size(S_mask)
        
        if size > best_size:
            best_size = size
            best_S_mask = S_mask
        
        seed += 1
    
    # Convert bitmask to list of 1-indexed integers
    result = [i + 1 for i in range(N) if (best_S_mask >> i) & 1]
    return result

# EVOLVE-BLOCK-END
