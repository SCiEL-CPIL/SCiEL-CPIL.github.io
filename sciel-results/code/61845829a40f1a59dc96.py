import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    prescribed = [1, 2, 5, 11]
    
    def is_valid_addition(x, current_set, mask):
        for y in current_set:
            s = x + y
            if (mask >> s) & 1:
                return False
        s = 2 * x
        if (mask >> s) & 1:
            return False
        return True

    def get_new_mask_bits(x, current_set):
        bits = 0
        for y in current_set:
            bits |= (1 << (x + y))
        bits |= (1 << (2 * x))
        return bits

    # Initialize mask with sums of prescribed elements
    initial_mask = 0
    for i in range(len(prescribed)):
        for j in range(i, len(prescribed)):
            s = prescribed[i] + prescribed[j]
            initial_mask |= (1 << s)

    best_set = prescribed[:]
    best_size = len(prescribed)
    target_size = 20

    # Randomized Greedy Search
    for restart in range(2000):
        if time.time() > deadline - 5:
            break
            
        current_set = prescribed[:]
        mask = initial_mask
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        
        while len(current_set) < target_size:
            if time.time() > deadline - 2:
                break
            
            added = False
            next_candidates = []
            added_idx = -1
            for i, x in enumerate(candidates):
                if is_valid_addition(x, current_set, mask):
                    current_set.append(x)
                    mask |= get_new_mask_bits(x, current_set[:-1])
                    added = True
                    added_idx = i
                    break
                else:
                    next_candidates.append(x)
            
            if not added:
                break
            
            # Update candidates: keep those before added_idx (already in next_candidates) 
            # and those after added_idx
            candidates = next_candidates + candidates[added_idx+1:]
            
            if len(current_set) > best_size:
                best_set = current_set[:]
                best_size = len(current_set)
                
            if best_size == target_size:
                return best_set

    return best_set
