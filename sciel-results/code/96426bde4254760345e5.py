import itertools
import random
import time
import math
import numpy as np

def construct(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls (neighbors within radius)
    balls = []
    for w in all_words:
        ball = [word_to_idx[w]]
        if radius == 1:
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    ball.append(word_to_idx[tuple(new_w)])
        balls.append(ball)
        
    # Initial Greedy Code
    def get_initial_code():
        uncovered = [1] * num_words
        num_unc = num_words
        code = []
        while num_unc > 0:
            best_c = -1
            best_cov = -1
            for c in range(num_words):
                cov = 0
                for u in balls[c]:
                    if uncovered[u]:
                        cov += 1
                if cov > best_cov:
                    best_cov = cov
                    best_c = c
            code.append(best_c)
            for u in balls[best_c]:
                if uncovered[u]:
                    uncovered[u] = 0
                    num_unc -= 1
        return code

    initial_code = get_initial_code()
    
    # State
    in_code = [False] * num_words
    cnt = [0] * num_words
    num_unc = 0
    
    for c in initial_code:
        in_code[c] = True
        for u in balls[c]:
            cnt[u] += 1
            
    for u in range(num_words):
        if cnt[u] == 0:
            num_unc += 1
            
    current_code_size = len(initial_code)
    best_code_size = current_code_size
    best_in_code = in_code[:]
    
    def objective(size, unc):
        return size + 10000 * unc

    current_obj = objective(current_code_size, num_unc)
    best_obj = current_obj
    
    start_time = time.time()
    time_limit = 110.0
    
    T = 50.0
    T_end = 0.1
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        move_type = random.random()
        
        if move_type < 0.7:
            # Try Remove
            c_to_remove = random.randint(0, num_words - 1)
            if not in_code[c_to_remove]:
                continue
            
            safe = True
            for u in balls[c_to_remove]:
                if cnt[u] <= 1:
                    safe = False
                    break
            
            if safe:
                in_code[c_to_remove] = False
                for u in balls[c_to_remove]:
                    cnt[u] -= 1
                current_code_size -= 1
                
                new_obj = objective(current_code_size, num_unc)
                delta = new_obj - current_obj
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_code_size = current_code_size
                    best_in_code = in_code[:]
                    
        elif move_type < 0.9:
            # Try Add
            c_to_add = random.randint(0, num_words - 1)
            if in_code[c_to_add]:
                continue
            
            in_code[c_to_add] = True
            for u in balls[c_to_add]:
                if cnt[u] == 0:
                    num_unc -= 1
                cnt[u] += 1
            current_code_size += 1
            
            new_obj = objective(current_code_size, num_unc)
            delta = new_obj - current_obj
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_code_size = current_code_size
                    best_in_code = in_code[:]
            else:
                in_code[c_to_add] = False
                for u in balls[c_to_add]:
                    cnt[u] -= 1
                    if cnt[u] == 0:
                        num_unc += 1
                current_code_size -= 1
                current_obj = objective(current_code_size, num_unc)
                
        else:
            # Try Swap
            c1 = random.randint(0, num_words - 1)
            if not in_code[c1]:
                continue
            c2 = random.randint(0, num_words - 1)
            if in_code[c2]:
                continue
            
            in_code[c1] = False
            for u in balls[c1]:
                if cnt[u] == 1:
                    num_unc += 1
                cnt[u] -= 1
            
            in_code[c2] = True
            for u in balls[c2]:
                if cnt[u] == 0:
                    num_unc -= 1
                cnt[u] += 1
            
            new_obj = objective(current_code_size, num_unc)
            delta = new_obj - current_obj
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_code_size = current_code_size
                    best_in_code = in_code[:]
            else:
                in_code[c2] = False
                for u in balls[c2]:
                    cnt[u] -= 1
                    if cnt[u] == 0:
                        num_unc += 1
                in_code[c1] = True
                for u in balls[c1]:
                    if cnt[u] == 0:
                        num_unc -= 1
                    cnt[u] += 1
                current_obj = objective(current_code_size, num_unc)

        T *= cooling
        if T < T_end:
            T = T_end

    result = []
    for i in range(num_words):
        if best_in_code[i]:
            result.append(list(all_words[i]))
    return result
