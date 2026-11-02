import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135  # 135s safety margin before 150s kill
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in product(range(q), repeat=n-4)]
        else:
            return [list(t) for t in product(range(q), repeat=n)]
    
    # Precompute all words
    all_words = np.array(list(product(range(3), repeat=6)), dtype=np.int8)
    N = 729
    
    # Precompute ball indices for each word
    word_to_idx = {}
    for i in range(N):
        word_to_idx[tuple(all_words[i])] = i
    
    balls = np.zeros((N, 19), dtype=np.int32)
    for i in range(N):
        w = all_words[i]
        balls[i, 0] = i
        count = 1
        for pos in range(6):
            for val in range(3):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    idx = word_to_idx[tuple(new_w)]
                    balls[i, count] = idx
                    count += 1
    
    # Precompute ball membership matrix for fast greedy
    ball_membership = np.zeros((N, N), dtype=bool)
    for i in range(N):
        ball_membership[i, balls[i]] = True
    
    # Greedy set cover
    uncovered_mask = np.ones(N, dtype=bool)
    codewords = []
    
    while uncovered_mask.any():
        if time.time() > deadline - 20:
            break
        coverage = (ball_membership & uncovered_mask[np.newaxis, :]).sum(axis=1)
        best_cw = int(np.argmax(coverage))
        if coverage[best_cw] == 0:
            break
        codewords.append(best_cw)
        uncovered_mask[balls[best_cw]] = False
    
    # Simulated Annealing with incremental coverage tracking
    code_list = list(codewords)
    
    # coverage_count[word] = number of codewords whose ball covers word
    coverage_count = np.zeros(N, dtype=np.int32)
    for cw in code_list:
        coverage_count[balls[cw]] += 1
    
    def is_fully_covered():
        return np.all(coverage_count > 0)
    
    def remove_cw(cw):
        coverage_count[balls[cw]] -= 1
    
    def add_cw(cw):
        coverage_count[balls[cw]] += 1
    
    def uncovered_count():
        return int(np.sum(coverage_count == 0))
    
    # SA parameters
    T_initial = 30.0
    T_min = 0.05
    cooling_rate = 0.997
    T = T_initial
    best_code = list(code_list)
    best_size = len(code_list)
    
    max_iterations = 50000
    iteration = 0
    
    while T > T_min and iteration < max_iterations:
        if time.time() > deadline - 15:
            break
        iteration += 1
        
        move_type = random.random()
        
        if len(code_list) > 1 and move_type < 0.6:
            # Try removal
            idx = random.randint(0, len(code_list) - 1)
            cw = code_list[idx]
            remove_cw(cw)
            
            if is_fully_covered():
                code_list.pop(idx)
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
            else:
                unc = uncovered_count()
                if random.random() < np.exp(-unc / T):
                    code_list.pop(idx)
                else:
                    add_cw(cw)
        
        elif len(code_list) > 1 and move_type < 0.85:
            # Try swap: remove one, add another
            idx = random.randint(0, len(code_list) - 1)
            cw_out = code_list[idx]
            in_code = set(code_list)
            candidates = [i for i in range(N) if i not in in_code]
            if candidates:
                cw_in = random.choice(candidates)
                remove_cw(cw_out)
                add_cw(cw_in)
                
                if is_fully_covered():
                    code_list[idx] = cw_in
                    if len(code_list) < best_size:
                        best_size = len(code_list)
                        best_code = list(code_list)
                else:
                    unc = uncovered_count()
                    if random.random() < np.exp(-unc / T):
                        code_list[idx] = cw_in
                    else:
                        remove_cw(cw_in)
                        add_cw(cw_out)
        
        else:
            # Try addition
            in_code = set(code_list)
            candidates = [i for i in range(N) if i not in in_code]
            if candidates:
                cw_in = random.choice(candidates)
                add_cw(cw_in)
                code_list.append(cw_in)
                if len(code_list) > best_size + 5:
                    code_list.pop()
                    remove_cw(cw_in)
        
        T *= cooling_rate
    
    # Use best found
    code_list = list(best_code)
    
    # Rebuild coverage_count for best_code
    coverage_count = np.zeros(N, dtype=np.int32)
    for cw in code_list:
        coverage_count[balls[cw]] += 1
    
    # Greedy removal pass
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        new_code_list = []
        for i, cw in enumerate(code_list):
            coverage_count[balls[cw]] -= 1
            if np.all(coverage_count > 0):
                improved = True
            else:
                new_code_list.append(cw)
                coverage_count[balls[cw]] += 1
        code_list = new_code_list
    
    # Final verification
    coverage_count = np.zeros(N, dtype=np.int32)
    for cw in code_list:
        coverage_count[balls[cw]] += 1
    
    if not np.all(coverage_count > 0):
        # Fallback to greedy result
        code_list = list(codewords)
    
    result = [list(all_words[cw]) for cw in code_list]
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
    
    return unique_result
