import random
import time

def score_set(S):
    S_set = set(S)
    good_d = set()
    for d in range(1, 512):
        for a in S:
            if (a + d) in S_set and (a + 2 * d) in S_set:
                good_d.add(d)
                break
    return len(good_d)

def run_search(initial_S, time_budget):
    start_time = time.time()
    S = initial_S.copy()
    current_score = score_set(S)
    best_S = S.copy()
    best_score = current_score
    
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_budget:
        new_S = S.copy()
        idx = random.randint(0, 31)
        old_val = new_S[idx]
        while True:
            new_val = random.randint(1, 512)
            if new_val not in new_S:
                break
        new_S[idx] = new_val
        
        new_score = score_set(new_S)
        
        if new_score > current_score:
            S = new_S
            current_score = new_score
            if current_score > best_score:
                best_S = S.copy()
                best_score = current_score
        else:
            delta = current_score - new_score
            if delta == 0:
                accept = True
            else:
                # Use exp(-delta/T)
                # delta is positive
                accept = random.random() < pow(2.71828, -delta / T)
            
            if accept:
                S = new_S
                current_score = new_score
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_S, best_score

def construct():
    start_time = time.time()
    total_time = 115
    
    # Run 1: Start from seed
    seed_S = list(range(1, 33))
    S1, score1 = run_search(seed_S, total_time / 3)
    
    # Run 2: Start from random
    random_S = random.sample(range(1, 513), 32)
    S2, score2 = run_search(random_S, total_time / 3)
    
    # Run 3: Start from another random
    random_S2 = random.sample(range(1, 513), 32)
    S3, score3 = run_search(random_S2, total_time / 3)
    
    # Pick best
    best_S = S1
    best_score = score1
    if score2 > best_score:
        best_S = S2
        best_score = score2
    if score3 > best_score:
        best_S = S3
        best_score = score3
        
    return best_S
