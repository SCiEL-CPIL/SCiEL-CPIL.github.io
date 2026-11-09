import time
import random
import math

TIME_LIMIT = 45.0

def can_add(x, mask, n=256):
    """
    Check if x can be added to the set represented by mask without creating a 3-AP.
    mask is an integer where bit i is set if i is in the set.
    We use 1-based indexing for values, but 0-based for bits? 
    Let's use bit (x-1) for value x.
    """
    if (mask >> (x - 1)) & 1:
        return False
    
    # Iterate through all y in the current set
    # To iterate bits, we can use:
    # m = mask
    # while m:
    #   low = m & -m
    #   y = low.bit_length()
    #   m ^= low
    
    m = mask
    while m:
        low = m & -m
        y = low.bit_length() # This gives 1-based index if we shift correctly? 
        # If bit 0 is set, low=1, bit_length=1. Value is 1.
        # If bit 1 is set, low=2, bit_length=2. Value is 2.
        # So y is the value.
        
        # Check 1: 2y - x in S
        c1 = 2 * y - x
        if c1 != x and 1 <= c1 <= n:
            if (mask >> (c1 - 1)) & 1:
                return False
        
        # Check 2: 2x - y in S
        c2 = 2 * x - y
        if c2 != y and 1 <= c2 <= n:
            if (mask >> (c2 - 1)) & 1:
                return False
        
        # Check 3: (x + y) / 2 in S
        s = x + y
        if s % 2 == 0:
            c3 = s // 2
            if c3 != x and c3 != y and 1 <= c3 <= n:
                if (mask >> (c3 - 1)) & 1:
                    return False
        
        m ^= low
        
    return True

def construct():
    start_time = time.time()
    n = 256
    recip = [0.0] * (n + 1)
    for i in range(1, n + 1):
        recip[i] = 1.0 / i
    
    # Greedy Initialization
    mask = 0
    current_score = 0.0
    for x in range(1, n + 1):
        if can_add(x, mask, n):
            mask |= (1 << (x - 1))
            current_score += recip[x]
    
    best_mask = mask
    best_score = current_score
    
    # Simulated Annealing
    temp = 0.05
    cooling = 0.9999
    min_temp = 1e-6
    
    # Helper to get list of elements from mask
    def mask_to_list(m):
        res = []
        while m:
            low = m & -m
            res.append(low.bit_length())
            m ^= low
        return res
    
    # Precompute available elements? 
    # We can just pick random x in 1..n and check if in mask.
    
    while time.time() - start_time < TIME_LIMIT:
        if temp < min_temp:
            break
            
        move_type = random.random()
        
        if move_type < 0.3:
            # Add
            x = random.randint(1, n)
            if not ((mask >> (x - 1)) & 1):
                if can_add(x, mask, n):
                    delta = recip[x]
                    new_mask = mask | (1 << (x - 1))
                    new_score = current_score + delta
                    if delta > 0 or random.random() < math.exp(delta / temp):
                        mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = mask
        
        elif move_type < 0.7:
            # Swap
            # Pick y in mask, x not in mask
            # To pick y in mask efficiently:
            # We can maintain a list of elements in the set? 
            # Or just pick random y and check if in mask, retry if not.
            # Since set is dense, this is fine.
            y = random.randint(1, n)
            if (mask >> (y - 1)) & 1:
                x = random.randint(1, n)
                if not ((mask >> (x - 1)) & 1):
                    # Check if removing y and adding x is valid
                    # This is equivalent to checking if x can be added to (mask without y)
                    # But can_add checks against mask. 
                    # We need to check if x creates AP with (S \ {y}).
                    # If x creates AP with y, that's fine because y is removed.
                    # If x creates AP with z in S\{y}, that's bad.
                    
                    # Let's construct the new mask and check? 
                    # Or modify can_add to take an exclusion?
                    # Simpler: 
                    # 1. Remove y from mask -> mask_temp
                    # 2. Check can_add(x, mask_temp)
                    mask_temp = mask & ~(1 << (y - 1))
                    if can_add(x, mask_temp, n):
                        delta = recip[x] - recip[y]
                        new_mask = mask_temp | (1 << (x - 1))
                        new_score = current_score + delta
                        if delta > 0 or random.random() < math.exp(delta / temp):
                            mask = new_mask
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_mask = mask
                            
        else:
            # Remove
            y = random.randint(1, n)
            if (mask >> (y - 1)) & 1:
                delta = -recip[y]
                new_mask = mask & ~(1 << (y - 1))
                new_score = current_score + delta
                # Always bad, so only accept if we are stuck? 
                # In SA, we accept bad moves to escape local minima.
                if random.random() < math.exp(delta / temp):
                    mask = new_mask
                    current_score = new_score
        
        temp *= cooling
        
        if time.time() - start_time > TIME_LIMIT - 2.0:
            break
            
    return mask_to_list(best_mask)

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(sum(1.0/x for x in res))
