import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    max_val = 256
    # Precompute LCMs? 
    # lcm(a,b) = a*b // gcd(a,b)
    
    best_score = 0
    best_set = [2]
    
    # We try each number as the "primary" anchor (smallest in set)
    # Actually, we don't know if the anchor is the smallest.
    # But if we iterate k from 2 to 256, and only consider x > k,
    # we cover all sets where k is the minimum element.
    
    for k in range(2, max_val + 1):
        if time.time() > deadline:
            break
            
        current_set = [k]
        current_score = 1.0 / k
        
        # Candidates must be > k
        # And compatible with k
        candidates = []
        for x in range(k + 1, max_val + 1):
            if math.lcm(k, x) > max_val:
                candidates.append(x)
        
        # Sort candidates by value (1/x) descending -> x ascending
        # candidates is already sorted by x ascending
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            # Check compatibility with all in current_set
            compatible = True
            for y in current_set:
                if math.lcm(x, y) <= max_val:
                    compatible = False
                    break
            
            if compatible:
                current_set.append(x)
                current_score += 1.0 / x
        
        if current_score > best_score:
            best_score = current_score
            best_set = current_set[:]
            
    return best_set
