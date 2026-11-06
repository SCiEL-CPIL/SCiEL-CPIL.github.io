import random
import time
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave buffer for 60s limit
    
    best_set = []
    best_size = 0
    
    # We will try to find a B_3 set in [1, 4096]
    # Max size is approx 42
    
    while time.time() - start_time < time_limit:
        current_set = []
        current_sums = set()
        
        # Random order of candidates
        candidates = list(range(1, 4097))
        random.shuffle(candidates)
        
        for x in candidates:
            # Generate new sums involving x
            # 1. x + x + x
            # 2. x + x + y for y in current_set
            # 3. x + y + z for y, z in current_set, y <= z
            
            new_sums = []
            new_sums.append(3 * x)
            
            for y in current_set:
                new_sums.append(2 * x + y)
                
            # y <= z pairs from current_set
            # Since current_set is small, simple nested loop is fine
            n = len(current_set)
            for i in range(n):
                for j in range(i, n):
                    new_sums.append(x + current_set[i] + current_set[j])
            
            # Check for duplicates within new_sums
            if len(set(new_sums)) != len(new_sums):
                continue
                
            # Check for collisions with existing sums
            if set(new_sums) & current_sums:
                continue
                
            # Valid, add x
            current_set.append(x)
            current_sums.update(new_sums)
            
            # Early termination if we can't beat best? 
            # Max possible remaining candidates is len(candidates) - current_index
            # But we don't know index easily in this loop structure without enumerate.
            # Just let it run.
            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
            # If we reach the theoretical max, we can stop
            # k^3/6 <= 3 * 4096 => k <= 41.9
            if best_size >= 42:
                break

    return best_set
