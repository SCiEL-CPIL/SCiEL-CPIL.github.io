import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 85.0
    max_val = 4096
    max_sum = 3 * max_val + 1
    
    def check_time():
        return time.time() - start_time > time_limit

    def greedy_find(target_size):
        """
        Greedily finds a set of size target_size.
        Tries candidates in random order to avoid bad local minima.
        """
        if target_size == 0:
            return []
        
        # Try a few random orders
        for _ in range(20):
            if check_time():
                break
            
            selected = []
            used_mask = 0
            
            # Randomize candidate order
            candidates = list(range(1, max_val + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                if len(selected) == target_size:
                    break
                
                k = len(selected)
                
                # Check if x can be added
                # New sums:
                # 3x
                # 2x + s for s in selected
                # x + s_i + s_j for i <= j in selected
                
                mask_x = 1 << (3*x)
                expected_bits = 1 + k + k*(k+1)//2
                valid = True
                
                # Check 3x
                if (used_mask >> (3*x)) & 1:
                    continue
                
                # Check 2x + s
                for s in selected:
                    mask_x |= 1 << (2*x + s)
                
                # Check x + s_i + s_j
                for i in range(k):
                    si = selected[i]
                    for j in range(i, k):
                        sj = selected[j]
                        mask_x |= 1 << (x + si + sj)
                
                # Check internal collisions and collision with used_mask
                if mask_x.bit_count() < expected_bits:
                    continue
                if mask_x & used_mask:
                    continue
                
                selected.append(x)
                used_mask |= mask_x
            
            if len(selected) == target_size:
                return selected
        
        return None

    # Theoretical max is 16.
    # Try to find 16.
    best_set = greedy_find(16)
    
    if best_set:
        return best_set
    
    # Fallback: Try 15, 14... just in case 16 is not found due to bad luck (very unlikely)
    for size in range(15, 0, -1):
        if check_time():
            break
        sol = greedy_find(size)
        if sol:
            return sol
            
    # Last resort: return empty or a small set
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] # Powers of 2 are B_h[1] for any h?
    # Check: 2^a + 2^b + 2^c. If a,b,c distinct, sum is unique.
    # If a=b, 2*2^a + 2^c = 2^{a+1} + 2^c. Unique.
    # If a=b=c, 3*2^a. Unique.
    # So powers of 2 work.
    # 1, 2, ..., 2048 is 11 elements.
    # We can add 4096? 4096 = 2^12.
    # So 12 elements.
    # Can we get 16 with powers of 2?
    # 2^0, ..., 2^15. Max val 32768 > 4096.
    # So powers of 2 only give us up to 12 elements in [1, 4096].
    
    # Better fallback: Greedy with sorted order
    selected = []
    used_mask = 0
    for x in range(1, max_val + 1):
        if check_time():
            break
        k = len(selected)
        mask_x = 1 << (3*x)
        expected_bits = 1 + k + k*(k+1)//2
        for s in selected:
            mask_x |= 1 << (2*x + s)
        for i in range(k):
            si = selected[i]
            for j in range(i, k):
                sj = selected[j]
                mask_x |= 1 << (x + si + sj)
        if mask_x.bit_count() < expected_bits:
            continue
        if mask_x & used_mask:
            continue
        selected.append(x)
        used_mask |= mask_x
        
    return selected
