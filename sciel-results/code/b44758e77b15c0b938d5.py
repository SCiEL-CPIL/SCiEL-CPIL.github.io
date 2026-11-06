import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    best_score = 0
    best_set = None
    all_nums = list(range(1, 513))
    
    for restart in range(300):
        if time.time() > deadline - 10:
            break
        
        # Initialize: random set of 32 elements from 1..512
        S = set(random.sample(all_nums, 32))
        S_list = list(S)
        S_bits = 0
        for x in S:
            S_bits |= (1 << x)
        
        cur_score = _score_set_bits(S_bits)
        
        if cur_score > best_score:
            best_score = cur_score
            best_set = set(S)
        
        # Local search with simulated annealing
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 5:
            if no_improve > max_no_improve:
                break
            
            # Choose move type: 50% single swap, 50% double swap
            if random.random() < 0.5:
                # Single swap
                x = random.choice(S_list)
                while True:
                    y = random.choice(all_nums)
                    if y not in S:
                        break
                
                S_bits_new = S_bits ^ (1 << x) ^ (1 << y)
                new_score = _score_set_bits(S_bits_new)
                
                delta = new_score - cur_score
                if delta > 0 or random.random() < np.exp(delta / T):
                    S.discard(x)
                    S.add(y)
                    S_list.remove(x)
                    S_list.append(y)
                    S_bits = S_bits_new
                    cur_score = new_score
                    no_improve = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_set = set(S)
                else:
                    no_improve += 1
            else:
                # Double swap
                x1, x2 = random.sample(S_list, 2)
                while True:
                    y1 = random.choice(all_nums)
                    if y1 in S:
                        continue
                    while True:
                        y2 = random.choice(all_nums)
                        if y2 in S or y2 == y1:
                            continue
                        break
                    break
                
                S_bits_new = S_bits ^ (1 << x1) ^ (1 << x2) ^ (1 << y1) ^ (1 << y2)
                new_score = _score_set_bits(S_bits_new)
                
                delta = new_score - cur_score
                if delta > 0 or random.random() < np.exp(delta / T):
                    S.discard(x1)
                    S.discard(x2)
                    S.add(y1)
                    S.add(y2)
                    S_list.remove(x1)
                    S_list.remove(x2)
                    S_list.append(y1)
                    S_list.append(y2)
                    S_bits = S_bits_new
                    cur_score = new_score
                    no_improve = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_set = set(S)
                else:
                    no_improve += 1
            
            T = max(T_min, T * cooling)
        
        # Polish phase: strict local search on single swaps
        polish_iter = 0
        while polish_iter < 20:
            if time.time() > deadline - 2:
                break
            best_swap_score = cur_score
            best_swap_x = -1
            best_swap_y = -1
            
            for i in range(len(S_list)):
                x = S_list[i]
                for y in all_nums:
                    if y in S:
                        continue
                    S_bits_new = S_bits ^ (1 << x) ^ (1 << y)
                    new_score = _score_set_bits(S_bits_new)
                    if new_score > best_swap_score:
                        best_swap_score = new_score
                        best_swap_x = x
                        best_swap_y = y
            
            if best_swap_x != -1:
                S.discard(best_swap_x)
                S.add(best_swap_y)
                S_list.remove(best_swap_x)
                S_list.append(best_swap_y)
                S_bits ^= (1 << best_swap_x) ^ (1 << best_swap_y)
                cur_score = best_swap_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_set = set(S)
                polish_iter += 1
            else:
                break
        
        T = 5.0
    
    if best_set is None:
        return list(range(1, 33))
    
    return list(best_set)

def _score_set_bits(bits):
    count = 0
    for d in range(1, 256):
        if (bits & (bits >> d) & (bits >> (2*d))) != 0:
            count += 1
    return count
