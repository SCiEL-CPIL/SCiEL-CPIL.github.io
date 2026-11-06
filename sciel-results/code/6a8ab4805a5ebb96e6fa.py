import random
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Initial set: 1 to 32
    current_set = set(range(1, 33))
    current_mask = 0
    for x in current_set:
        current_mask |= (1 << x)
    
    current_score = _score_mask(current_mask)
    best_set = set(current_set)
    best_mask = current_mask
    best_score = current_score
    
    T = 5.0
    cooling = 0.9995
    min_T = 0.01

    def _score_mask(mask):
        count = 0
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> 2*d)) != 0:
                count += 1
        return count

    iterations = 0
    while time.time() < deadline:
        if time.time() > deadline - 5:
            break
            
        # Pick an element to remove
        # To speed up, we can pick a random index from the set
        # Converting set to list every time is O(N). 
        # We can maintain a list representation? 
        # Or just use random.sample.
        
        # Let's stick to simple set operations for correctness, 
        # but optimize the swap.
        
        items = list(current_set)
        old_val = items[random.randint(0, len(items)-1)]
        
        while True:
            new_val = random.randint(1, 512)
            if new_val not in current_set:
                break
        
        # Apply swap to mask
        new_mask = (current_mask & ~(1 << old_val)) | (1 << new_val)
        
        new_score = _score_mask(new_mask)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            # Accept
            current_set.remove(old_val)
            current_set.add(new_val)
            current_mask = new_mask
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
                best_mask = current_mask
        
        T *= cooling
        if T < min_T:
            T = min_T
            
        iterations += 1
        if iterations % 500 == 0 and time.time() > deadline - 10:
            break

    return list(best_set)
# EVOLVE-BLOCK-END
