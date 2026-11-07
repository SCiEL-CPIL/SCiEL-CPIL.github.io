import time
import random

def construct():
    deadline = time.time() + 85
    
    best_size = 0
    best_set = []
    
    candidates = list(range(1, 257))
    
    # Fixed orderings
    orderings = []
    orderings.append(list(range(1, 257)))
    orderings.append(list(range(256, 0, -1)))
    orderings.append([x for x in range(1, 257, 2)] + [x for x in range(2, 257, 2)])
    orderings.append([x for x in range(2, 257, 2)] + [x for x in range(1, 257, 2)])
    alt = []
    for i in range(128):
        alt.append(1 + i)
        alt.append(129 + i)
    orderings.append(alt)
    
    # Process fixed orderings
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        current_set = []
        max_possible_sum = 60 * 256
        sum_to_card = [-1] * (max_possible_sum + 1)
        sum_to_card[0] = 0
        active_sums = [0]
        n = 0
        
        for x in ordering:
            if n >= 60:
                break
            
            valid = True
            new_sums = []
            for s in active_sums:
                ns = s + x
                nk = sum_to_card[s] + 1
                if sum_to_card[ns] >= 0 and sum_to_card[ns] != nk:
                    valid = False
                    break
                new_sums.append((ns, nk))
            
            if not valid:
                continue
            
            for ns, nk in new_sums:
                if sum_to_card[ns] < 0:
                    sum_to_card[ns] = nk
                    active_sums.append(ns)
            
            current_set.append(x)
            n += 1
        
        if n > best_size:
            best_size = n
            best_set = current_set[:]
    
    # Use remaining time for random shuffles
    while time.time() < deadline - 2: # Safety margin
        o = candidates[:]
        random.shuffle(o)
        
        current_set = []
        max_possible_sum = 60 * 256
        sum_to_card = [-1] * (max_possible_sum + 1)
        sum_to_card[0] = 0
        active_sums = [0]
        n = 0
        
        for x in o:
            if n >= 60:
                break
            
            valid = True
            new_sums = []
            for s in active_sums:
                ns = s + x
                nk = sum_to_card[s] + 1
                if sum_to_card[ns] >= 0 and sum_to_card[ns] != nk:
                    valid = False
                    break
                new_sums.append((ns, nk))
            
            if not valid:
                continue
            
            for ns, nk in new_sums:
                if sum_to_card[ns] < 0:
                    sum_to_card[ns] = nk
                    active_sums.append(ns)
            
            current_set.append(x)
            n += 1
        
        if n > best_size:
            best_size = n
            best_set = current_set[:]
    
    return best_set
