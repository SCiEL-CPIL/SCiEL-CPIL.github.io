import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 128
    
    # Precompute for each x:
    # case1_mask[x]: bitmask of a's such that a+x <= n (i.e., a in 1..n-x)
    # case2_pairs[x]: list of (a,b) with a<b, a+b=x, both in 1..n
    # For case 2, we also precompute the combined mask for each pair
    
    # Precompute case1: for each x, the mask of valid a positions (a in 1..n-x)
    case1_a_mask = [0] * (n + 1)
    for x in range(1, n + 1):
        if x <= n:
            case1_a_mask[x] = (1 << (n - x)) - 1  # bits 0..n-x-1 set
    
    # Precompute case2: for each x, list of (a, b, c_mask) where a+b=x, a<b
    # c_mask[a,b] = mask of c's such that a+c<=n and b+c<=n, excluding a and b
    # Actually for case 2: C_mask = (mask & (mask >> a)) & (mask & (mask >> b)) & ~(1<<(a-1)) & ~(1<<(b-1))
    # We can precompute the exclusion mask for each pair
    case2_data = [None] * (n + 1)
    for x in range(3, n + 1):
        pairs = []
        for a in range(1, x // 2 + 1):
            b = x - a
            if b > a:
                # Precompute the exclusion bits
                excl = (1 << (a - 1)) | (1 << (b - 1))
                pairs.append((a, b, excl))
        case2_data[x] = pairs
    
    def can_add(mask, x):
        """Check if x can be added to set represented by mask. x is 1-indexed."""
        bit_x = 1 << (x - 1)
        if mask & bit_x:
            return True
        
        # Case 1: x is the third vertex c.
        # mask_a = bits where a is in S and a+x is in S, restricted to a in 1..n-x
        mask_a = (mask & (mask >> x)) & case1_a_mask[x]
        # Clear bit for a=x if applicable
        if x - 1 < n - x:
            mask_a &= ~(1 << (x - 1))
        
        if mask_a & (mask_a - 1):  # at least 2 bits set
            # For each pair (a,b) in mask_a, check if a+b in S or a+b == x
            # Iterate over set bits
            a_list = []
            m = mask_a
            while m:
                lb = m & (-m)
                a_list.append(lb.bit_length())
                m ^= lb
            
            if len(a_list) >= 2:
                for i in range(len(a_list)):
                    a = a_list[i]
                    for j in range(len(a_list)):
                        if i == j:
                            continue
                        b = a_list[j]
                        s = a + b
                        if s == x:
                            return False
                        if s <= n and (mask >> (s - 1)) & 1:
                            return False
        
        # Case 2: x = a + b
        if case2_data[x]:
            for a, b, excl in case2_data[x]:
                if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1:
                    C_mask = (mask & (mask >> a)) & (mask & (mask >> b)) & ~excl
                    if C_mask != 0:
                        return False
        
        return True
    
    def popcount(m):
        return bin(m).count('1')
    
    def simulated_annealing(mask, time_limit):
        """Simulated annealing local search."""
        best = mask
        best_size = popcount(mask)
        current_size = best_size
        t = time.time()
        end_t = t + time_limit
        
        # Temperature schedule
        T0 = 0.5
        T_end = 0.01
        
        while time.time() < end_t:
            elapsed_frac = (time.time() - t) / max(time_limit, 1e-9)
            T = T0 * (T_end / T0) ** elapsed_frac
            
            # Choose move type
            r = random.random()
            if r < 0.6:
                # Add a random missing element
                missing_bits = ((1 << n) - 1) & ~mask
                if missing_bits:
                    # Pick a random set bit
                    # Use random selection among set bits
                    idx = random.randrange(n)
                    # Find a set bit: use a simple method
                    m = missing_bits
                    # Pick random position among set bits
                    count = popcount(m)
                    if count > 0:
                        target = random.randrange(count)
                        mm = m
                        while mm:
                            lb = mm & (-mm)
                            if target == 0:
                                x = lb.bit_length()
                                break
                            target -= 1
                            mm ^= lb
                        if can_add(mask, x):
                            mask |= (1 << (x - 1))
                            current_size += 1
                            if current_size > best_size:
                                best_size = current_size
                                best = mask
            elif r < 0.8:
                # Remove a random present element
                present_bits = mask
                if present_bits:
                    count = popcount(present_bits)
                    if count > 0:
                        target = random.randrange(count)
                        mm = present_bits
                        while mm:
                            lb = mm & (-mm)
                            if target == 0:
                                x = lb.bit_length()
                                break
                            target -= 1
                            mm ^= lb
                        mask &= ~(1 << (x - 1))
                        current_size -= 1
            else:
                # Swap: remove one, add another
                present_bits = mask
                missing_bits = ((1 << n) - 1) & ~mask
                if present_bits and missing_bits:
                    count_p = popcount(present_bits)
                    count_m = popcount(missing_bits)
                    if count_p > 0 and count_m > 0:
                        # Pick element to remove
                        target = random.randrange(count_p)
                        mm = present_bits
                        while mm:
                            lb = mm & (-mm)
                            if target == 0:
                                x_rem = lb.bit_length()
                                break
                            target -= 1
                            mm ^= lb
                        # Pick element to add
                        target = random.randrange(count_m)
                        mm = missing_bits
                        while mm:
                            lb = mm & (-mm)
                            if target == 0:
                                x_add = lb.bit_length()
                                break
                            target -= 1
                            mm ^= lb
                        # Try the swap
                        new_mask = (mask & ~(1 << (x_rem - 1))) | (1 << (x_add - 1))
                        if can_add(new_mask, x_add):
                            # Accept with probability based on size change
                            delta = 0  # size stays same in swap
                            # Actually swap keeps size same, so always accept if valid
                            mask = new_mask
                            # current_size unchanged
        
        # Final greedy fill
        for x in range(1, n + 1):
            if not (mask >> (x - 1)) & 1:
                if can_add(mask, x):
                    mask |= (1 << (x - 1))
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best = mask
        
        return best
    
    best_mask = 0
    best_size = 0
    
    # Multiple restarts
    remaining = deadline - time.time()
    num_restarts = max(5, int(remaining / 2.0))
    
    for r in range(num_restarts):
        if time.time() >= deadline - 3:
            break
        
        # Random ordering for greedy init
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        # Greedy MIS
        mask = 0
        for x in order:
            if can_add(mask, x):
                mask |= (1 << (x - 1))
        
        sz = popcount(mask)
        
        # Simulated annealing with remaining time budget
        sa_time = (deadline - time.time()) / max(1, num_restarts - r) - 0.5
        if sa_time > 0.2:
            mask = simulated_annealing(mask, sa_time)
            sz = popcount(mask)
        
        if sz > best_size:
            best_size = sz
            best_mask = mask
    
    # Final greedy pass on best
    for x in range(1, n + 1):
        if not (best_mask >> (x - 1)) & 1:
            if can_add(best_mask, x):
                best_mask |= (1 << (x - 1))
    
    # Convert to list
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    
    # Final verification
    result_set = set(result)
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            for k in range(j + 1, len(result)):
                a, b, c = result[i], result[j], result[k]
                s1, s2, s3 = a + b, a + c, b + c
                if s1 in result_set and s2 in result_set and s3 in result_set:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return list(range(65, 129))
    
    return result
