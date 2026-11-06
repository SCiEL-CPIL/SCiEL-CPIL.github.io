import time
import random
import math

def _score_bitset(B):
    count = 0
    for d in range(1, 256):
        if (B & (B >> d) & (B >> (2*d))) != 0:
            count += 1
    return count

def _bits_to_list(B):
    return [i for i in range(512) if (B >> i) & 1]

def _greedy_construct():
    B = 0
    for _ in range(32):
        best_x = -1
        best_score = -1
        for x in range(1, 513):
            if (B >> x) & 1:
                continue
            new_B = B | (1 << x)
            s = _score_bitset(new_B)
            if s > best_score:
                best_score = s
                best_x = x
        if best_x == -1:
            break
        B |= (1 << best_x)
    return B

def _simulated_annealing(B, time_limit):
    current_B = B
    current_score = _score_bitset(current_B)
    best_B = current_B
    best_score = current_score
    
    elements = _bits_to_list(current_B)
    
    T = 5.0
    T_min = 0.01
    alpha = 0.9995
    
    start_time = time.time()
    
    while time.time() < time_limit:
        idx = random.randint(0, len(elements) - 1)
        x = elements[idx]
        
        y = random.randint(1, 512)
        while (current_B >> y) & 1:
            y = random.randint(1, 512)
        
        new_B = (current_B & ~(1 << x)) | (1 << y)
        new_score = _score_bitset(new_B)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_B = new_B
            current_score = new_score
            
            elements[idx] = y
            
            if current_score > best_score:
                best_score = current_score
                best_B = current_B
        
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_B, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Greedy Initialization
    B_greedy = _greedy_construct()
    best_B = B_greedy
    best_score = _score_bitset(B_greedy)
    
    # 2. Simulated Annealing
    B_sa, score_sa = _simulated_annealing(B_greedy, deadline)
    if score_sa > best_score:
        best_B = B_sa
        best_score = score_sa
        
    # 3. If time remains, re-anneal from best
    if time.time() < deadline - 5.0:
        B_sa2, score_sa2 = _simulated_annealing(best_B, deadline)
        if score_sa2 > best_score:
            best_B = B_sa2
            best_score = score_sa2

    return _bits_to_list(best_B)
