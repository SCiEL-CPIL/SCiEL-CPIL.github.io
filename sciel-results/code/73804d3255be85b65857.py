import time
import random
import numpy as np

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    def evaluate(S):
        S_set = set(S)
        valid_diffs = set()
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                d = b - a
                c = a + 2 * d
                if c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)
    
    # Known good initial set from parent (score 4.8)
    initial_set = set(list(range(1, 27)) + [28, 29, 30, 31, 32, 48])
    
    def random_set():
        return set(random.sample(range(1, 513), 32))
    
    def structured_set():
        base = random.randint(1, 400)
        s = set()
        block_size = random.randint(8, 16)
        for i in range(block_size):
            val = base + i * random.randint(1, 3)
            if 1 <= val <= 512 and val not in s:
                s.add(val)
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        return s
    
    best_set = None
    best_score = 0
    
    candidates = [
        initial_set,
        set(range(1, 33)),
        random_set(),
        random_set(),
        structured_set(),
        structured_set(),
    ]
    
    for c in candidates:
        s = evaluate(c)
        if s > best_score:
            best_score = s
            best_set = c.copy()
    
    current_set = best_set.copy()
    current_score = best_score
    
    no_improvement_count = 0
    all_elements = list(range(1, 513))
    
    while time.time() - start_time < timeout:
        improved = False
        steps_in_iteration = 0
        max_steps_per_iter = 3000
        
        while steps_in_iteration < max_steps_per_iter:
            if time.time() - start_time > timeout - 5.0:
                break
            
            move_type = random.random()
            new_set = set(current_set)
            
            if move_type < 0.6:
                remove_idx = random.randint(0, 31)
                x_to_remove = list(current_set)[remove_idx]
                y_to_add = random.choice(all_elements)
                if y_to_add in current_set:
                    steps_in_iteration += 1
                    continue
                new_set.remove(x_to_remove)
                new_set.add(y_to_add)
            else:
                remove_idx1 = random.randint(0, 31)
                remove_idx2 = random.randint(0, 31)
                if remove_idx1 == remove_idx2:
                    steps_in_iteration += 1
                    continue
                x1 = list(current_set)[remove_idx1]
                x2 = list(current_set)[remove_idx2]
                y1 = random.choice(all_elements)
                y2 = random.choice(all_elements)
                if y1 in current_set or y2 in current_set:
                    steps_in_iteration += 1
                    continue
                new_set.remove(x1)
                new_set.remove(x2)
                new_set.add(y1)
                new_set.add(y2)
            
            new_score = evaluate(new_set)
            
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                improved = True
                no_improvement_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set.copy()
            else:
                no_improvement_count += 1
            
            steps_in_iteration += 1
        
        if time.time() - start_time > timeout - 5.0:
            break
        
        if no_improvement_count > 200:
            num_swaps = random.randint(3, 8)
            for _ in range(num_swaps):
                remove_idx = random.randint(0, 31)
                x_to_remove = list(current_set)[remove_idx]
                y_to_add = random.choice(all_elements)
                if y_to_add not in current_set:
                    current_set.remove(x_to_remove)
                    current_set.add(y_to_add)
            current_score = evaluate(current_set)
            no_improvement_count = 0
        
        if no_improvement_count > 1000:
            current_set = best_set.copy()
            current_score = best_score
            no_improvement_count = 0

    return list(best_set)
