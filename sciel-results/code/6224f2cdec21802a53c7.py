# EVOLVE-BLOCK-START
import time
import random
import math

N = 256
MAX_D_3AP = (N - 1) // 2
MAX_D_4AP = (N - 1) // 3

def _count_3aps(mask):
    count = 0
    for d in range(1, MAX_D_3AP + 1):
        count += (mask & (mask >> d) & (mask >> (2 * d))).bit_count()
    return count

def _has_4ap(mask):
    for d in range(1, MAX_D_4AP + 1):
        if (mask & (mask >> d) & (mask >> (2 * d)) & (mask >> (3 * d))) != 0:
            return True
    return False

def _greedy_construct():
    mask = 0
    for x in range(1, N + 1):
        bit = 1 << (x - 1)
        new_mask = mask | bit
        if not _has_4ap(new_mask):
            mask = new_mask
    return mask

def _random_greedy():
    mask = 0
    elements = list(range(1, N + 1))
    random.shuffle(elements)
    for x in elements:
        bit = 1 << (x - 1)
        new_mask = mask | bit
        if not _has_4ap(new_mask):
            mask = new_mask
    return mask

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_mask = _greedy_construct()
    best_score = _count_3aps(best_mask)
    
    T = 1.0
    cooling_rate = 0.9995
    min_T = 1e-5
    
    restarts = 0
    max_restarts = 200
    
    while time.time() < deadline and restarts < max_restarts:
        if restarts == 0:
            current_mask = best_mask
        elif restarts % 5 == 0:
            current_mask = _random_greedy()
        else:
            current_mask = best_mask
            
        current_score = _count_3aps(current_mask)
        
        while T > min_T and time.time() < deadline:
            move_type = random.random()
            if move_type < 0.4:
                idx = random.randint(0, N - 1)
                if not (current_mask & (1 << idx)):
                    new_mask = current_mask | (1 << idx)
                    if _has_4ap(new_mask):
                        continue
                    new_score = _count_3aps(new_mask)
                    delta = new_score - current_score
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
            elif move_type < 0.8:
                idx = random.randint(0, N - 1)
                if current_mask & (1 << idx):
                    new_mask = current_mask & ~(1 << idx)
                    new_score = _count_3aps(new_mask)
                    delta = new_score - current_score
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
            else:
                idx_in = random.randint(0, N - 1)
                idx_out = random.randint(0, N - 1)
                if (current_mask & (1 << idx_in)) == 0 and (current_mask & (1 << idx_out)):
                    new_mask = (current_mask | (1 << idx_in)) & ~(1 << idx_out)
                    if _has_4ap(new_mask):
                        continue
                    new_score = _count_3aps(new_mask)
                    delta = new_score - current_score
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
            
            T *= cooling_rate
        
        restarts += 1
    
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
    
    return result
# EVOLVE-BLOCK-END
