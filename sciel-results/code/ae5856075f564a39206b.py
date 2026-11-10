import time
import random

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin for 60s limit
    
    # Known valid set of size 31
    fallback = [226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256]
    
    # Try to find a set of size 32
    # Initialize with AP
    current_set = list(range(225, 257)) # 225..256
    
    def min_gap(s):
        s = sorted(s)
        n = len(s)
        min_g = float('inf')
        for k in range(1, n):
            # Max sum of k largest
            max_sum_k = sum(s[n-k:])
            # Min sum of k+1 smallest
            min_sum_k1 = sum(s[:k+1])
            gap = min_sum_k1 - max_sum_k
            if gap < min_g:
                min_g = gap
        return min_g

    best_gap = min_gap(current_set)
    best_set = current_set[:]
    
    # Local search
    # We want to increase min_gap
    # Perturb elements
    # To increase min_sum_k1, increase small elements
    # To decrease max_sum_k, decrease large elements
    
    random.seed(42)
    
    for i in range(5000):
        if time.time() > deadline:
            break
        
        # Pick a random element to perturb
        idx = random.randint(0, 31)
        # Direction: if idx is in lower half, increase; if in upper half, decrease
        # This compresses the set
        if idx < 16:
            delta = random.choice([1, 2, 3])
        else:
            delta = random.choice([-1, -2, -3])
            
        new_val = current_set[idx] + delta
        if 1 <= new_val <= 256:
            # Check distinctness
            if new_val not in current_set:
                old_val = current_set[idx]
                current_set[idx] = new_val
                g = min_gap(current_set)
                if g > best_gap:
                    best_gap = g
                    best_set = current_set[:]
                    if best_gap > 0:
                        # Found valid set
                        return best_set
                else:
                    # Revert
                    current_set[idx] = old_val

    # If we found a set with gap > 0, return it
    if best_gap > 0:
        return best_set
    
    # Otherwise return fallback
    return fallback
