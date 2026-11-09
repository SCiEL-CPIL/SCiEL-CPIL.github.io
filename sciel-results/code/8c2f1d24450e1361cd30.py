import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 128
    all_elements = list(range(1, n + 1))
    
    def popcount(mask):
        return bin(mask).count('1')
    
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

    best_mask = 0
    best_size = 0
    
    # Strategy: Multiple randomized greedy restarts
    # This explores different maximal independent sets quickly.
    
    # 1. Try a few fixed strategies
    fixed_orders = [
        list(range(1, n+1)),
        list(range(n, 0, -1)),
        list(range(65, n+1)) + list(range(1, 65)),
        list(range(85, n+1)) + list(range(1, 85))
    ]
    
    for order in fixed_orders:
        if time.time() > deadline - 2:
            break
        m = randomized_greedy(order)
        m = polish(m)
        s = popcount(m)
        if s > best_size:
            best_mask = m
            best_size = s

    # 2. Random restarts
    # Shuffle the order and run greedy
    while time.time() < deadline - 2:
        order = list(range(1, n+1))
        random.shuffle(order)
        m = randomized_greedy(order)
        m = polish(m)
        s = popcount(m)
        if s > best_size:
            best_mask = m
            best_size = s
    
    # Final polish on the best mask (though polish is already applied)
    # Maybe try a different polish order?
    if time.time() < deadline - 1:
        # Try to add elements in decreasing order?
        # The polish function uses increasing order.
        # Let's just ensure no simple additions are left.
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
