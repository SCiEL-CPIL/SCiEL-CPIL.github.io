import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 128
    all_elements = list(range(1, n + 1))
    
    def popcount(mask):
        return mask.bit_count()
    
    def is_valid_addition_fast(mask, x):
        """
        Fast check: can we add x to the set represented by mask?
        Returns True if valid.
        """
        bit_x = 1 << (x - 1)
        if mask & bit_x:
            return True
        
        # Case 1: x is a vertex c. Need a,b in S with a+x in S, b+x in S, and a+b in S (or a+b==x)
        C_a_mask = mask & (mask >> x)
        
        if C_a_mask.bit_count() >= 2:
            # Build bitmask of candidates (a in S such that a+x in S, a != x)
            cand_mask = 0
            temp = C_a_mask
            while temp:
                low_bit = temp & (-temp)
                idx = low_bit.bit_length() - 1
                a = idx + 1
                if a != x:
                    cand_mask |= (1 << (a-1))
                temp ^= low_bit
            
            # Check if any pair (a,b) in cand_mask sums to an element in mask
            temp = cand_mask
            while temp:
                low_bit = temp & (-temp)
                idx = low_bit.bit_length() - 1
                a = idx + 1
                # (mask >> a) has bit (b-1) set if a+b in mask
                # We need b in cand_mask, b != a
                if (cand_mask & ~(1 << (a-1)) & (mask >> a)):
                    return False
                temp ^= low_bit
        
        # Case 2: x is a sum. x = a+b for some a,b in S.
        for a in range(1, x//2 + 1):
            b = x - a
            if b <= a:
                continue
            if (mask >> (a-1)) & 1 and (mask >> (b-1)) & 1:
                C_a = mask & (mask >> a)
                C_b = mask & (mask >> b)
                potential_c = C_a & C_b
                potential_c &= ~(1 << (a-1))
                potential_c &= ~(1 << (b-1))
                if potential_c != 0:
                    return False
        
        return True
    
    def can_add_to_mask(mask, x):
        """Check if x can be added to mask without violating constraints."""
        return is_valid_addition_fast(mask, x)
    
    def remove_from_mask(mask, x):
        """Remove x from mask."""
        return mask & ~(1 << (x - 1))
    
    def try_add_to_mask(mask, x):
        """Try to add x to mask, return new mask if valid else None."""
        if mask & (1 << (x - 1)):
            return mask
        if is_valid_addition_fast(mask, x):
            return mask | (1 << (x - 1))
        return None
    
    def randomized_greedy(order):
        """
        Construct a set by iterating through 'order' and adding valid elements.
        """
        mask = 0
        for x in order:
            if is_valid_addition_fast(mask, x):
                mask |= (1 << (x-1))
        return mask

    def polish(mask):
        """
        Try to add any remaining valid elements in increasing order.
        """
        for x in range(1, n+1):
            if not (mask >> (x-1)) & 1:
                if is_valid_addition_fast(mask, x):
                    mask |= (1 << (x-1))
        return mask

    def local_search(mask, max_iters=500):
        """
        Local search: try to swap out one element for one or more others.
        This can escape local optima of pure greedy.
        """
        current_mask = mask
        current_size = current_mask.bit_count()
        best_mask = current_mask
        best_size = current_size
        no_improve_count = 0
        
        for iteration in range(max_iters):
            if time.time() > deadline - 3:
                break
            if no_improve_count > 100:
                break
            
            # Get list of elements in current set
            elements = []
            temp = current_mask
            while temp:
                low_bit = temp & (-temp)
                idx = low_bit.bit_length() - 1
                elements.append(idx + 1)
                temp ^= low_bit
            
            if not elements:
                break
            
            # Try swapping out one random element
            removed = random.choice(elements)
            new_mask = current_mask & ~(1 << (removed - 1))
            
            # Try to add back other elements in random order
            other_elements = [x for x in range(1, n+1) if x != removed]
            random.shuffle(other_elements)
            
            improved = False
            for x in other_elements:
                if not (new_mask >> (x-1)) & 1:
                    if is_valid_addition_fast(new_mask, x):
                        new_mask |= (1 << (x-1))
                        improved = True
                        if new_mask.bit_count() > best_size:
                            best_mask = new_mask
                            best_size = new_mask.bit_count()
            
            new_size = new_mask.bit_count()
            if new_size > current_size:
                current_mask = new_mask
                current_size = new_size
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        return best_mask, best_size

    best_mask = 0
    best_size = 0
    
    # Strategy: Multiple randomized greedy restarts with local search
    # This explores different maximal independent sets quickly.
    
    # 1. Try a few fixed strategies
    fixed_orders = [
        list(range(1, n+1)),
        list(range(n, 0, -1)),
        list(range(65, n+1)) + list(range(1, 65)),
        list(range(85, n+1)) + list(range(1, 85)),
        list(range(1, n+1, 2)),  # odds first
        list(range(2, n+1, 2)),  # evens first
    ]
    
    for order in fixed_orders:
        if time.time() > deadline - 3:
            break
        m = randomized_greedy(order)
        m = polish(m)
        s = popcount(m)
        if s > best_size:
            best_mask = m
            best_size = s

    # 2. Random restarts with local search
    restart_count = 0
    while time.time() < deadline - 5:
        order = list(range(1, n+1))
        random.shuffle(order)
        m = randomized_greedy(order)
        m = polish(m)
        s = popcount(m)
        
        # Apply local search to improve
        m_ls, s_ls = local_search(m, max_iters=200)
        
        if s_ls > best_size:
            best_mask = m_ls
            best_size = s_ls
        
        restart_count += 1
        # Increase local search iterations for later restarts if time permits
        if restart_count % 10 == 0 and time.time() < deadline - 8:
            # Try more aggressive local search on best so far
            m_ls2, s_ls2 = local_search(best_mask, max_iters=300)
            if s_ls2 > best_size:
                best_mask = m_ls2
                best_size = s_ls2
    
    # Final polish on the best mask
    if time.time() < deadline - 1:
        # Try to add elements in increasing order
        for x in range(1, n+1):
            if not (best_mask >> (x-1)) & 1:
                if is_valid_addition_fast(best_mask, x):
                    best_mask |= (1 << (x-1))
                    best_size += 1
        # Try to add elements in decreasing order
        for x in range(n, 0, -1):
            if not (best_mask >> (x-1)) & 1:
                if is_valid_addition_fast(best_mask, x):
                    best_mask |= (1 << (x-1))
                    best_size += 1

    # Convert to list
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    
    return result
