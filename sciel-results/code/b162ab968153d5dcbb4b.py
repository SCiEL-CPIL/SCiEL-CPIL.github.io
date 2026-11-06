import random
import time
import math

def _get_dense_seeds():
    seeds = []
    for x in range(1, 513):
        c_first = (512 - x) // 2
        c_mid = min(x - 1, 512 - x)
        c_last = (x - 1) // 2
        total = c_first + c_mid + c_last
        seeds.append((total, x))
    seeds.sort(reverse=True)
    return [x for _, x in seeds]

def _calc_score(mask):
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def construct():
    all_dense = _get_dense_seeds()
    initial_indices = all_dense[:32]
    
    mask = 0
    for idx in initial_indices:
        mask |= (1 << (idx - 1))
    
    current_score = _calc_score(mask)
    best_mask = mask
    best_score = current_score
    
    current_list = list(initial_indices)
    current_set = set(initial_indices)
    
    start_time = time.time()
    time_limit = 110
    
    temp = 10.0
    cooling = 0.999
    min_temp = 0.1
    
    all_indices = list(range(512))
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        idx = random.randrange(32)
        to_remove = current_list[idx]
        
        to_add = random.choice(all_indices) + 1
        
        if to_add in current_set:
            continue
            
        current_list[idx] = to_add
        current_set.discard(to_remove)
        current_set.add(to_add)
        
        mask &= ~(1 << (to_remove - 1))
        mask |= (1 << (to_add - 1))
        
        new_score = _calc_score(mask)
        delta = new_score - current_score
        
        if delta >= 0 or random.random() < math.exp(delta / temp):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = mask
        else:
            # Revert
            current_list[idx] = to_remove
            current_set.discard(to_add)
            current_set.add(to_remove)
            
            mask &= ~(1 << (to_add - 1))
            mask |= (1 << (to_remove - 1))
            
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
            
    result = []
    for i in range(512):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
