import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    all_nums = list(range(1, 513))
    
    def score_set(S):
        bits = 0
        for x in S:
            bits |= (1 << x)
        count = 0
        for d in range(1, 256):
            if (bits & (bits >> d) & (bits >> (2*d))) != 0:
                count += 1
        return count

    # Candidate initializations
    candidates = []
    
    # 1. Dense core + 1 outlier
    candidates.append(set(list(range(1, 32)) + [47]))
    
    # 2. Dense core + 2 outliers
    candidates.append(set(list(range(1, 31)) + [47, 53]))
    
    # 3. Modified core to cover 15, 16
    candidates.append(set(list(range(1, 30)) + [31, 47, 53]))
    
    # 4. Random
    candidates.append(set(random.sample(all_nums, 32)))
    
    best_score = 0
    best_set = None
    
    for S in candidates:
        s = score_set(S)
        if s > best_score:
            best_score = s
            best_set = set(S)
            
    # Local Search
    S = best_set
    cur_score = best_score
    S_list = list(S)
    
    T = 2.0
    T_min = 0.1
    cooling = 0.99
    
    while time.time() < deadline - 5:
        # Pick a move
        x = random.choice(S_list)
        y = random.choice(all_nums)
        if y in S:
            continue
            
        S_new = set(S)
        S_new.remove(x)
        S_new.add(y)
        
        new_score = score_set(S_new)
        
        delta = new_score - cur_score
        if delta > 0 or random.random() < pow(2.718281828, delta / T):
            S = S_new
            S_list = list(S)
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_set = set(S)
        
        T = max(T_min, T * cooling)
        
    return list(best_set)
