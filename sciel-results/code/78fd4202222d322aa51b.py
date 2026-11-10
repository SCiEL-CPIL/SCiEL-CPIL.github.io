import time
import random
import numpy as np

def _compute_cost(A):
    A = list(A)
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))

def _generate_initial_candidate():
    candidates = []
    candidates.append(list(range(1, 21)))
    
    # GP truncated
    gp = [2**i for i in range(14)]
    remaining = [x for x in range(1, 10001) if x not in gp]
    random.shuffle(remaining)
    gp.extend(remaining[:6])
    random.shuffle(gp)
    candidates.append(sorted(gp))
    
    # Random
    candidates.append(sorted(random.sample(range(1, 10001), 20)))
    
    best_c = None
    best_cost = float('inf')
    for c in candidates:
        cost = _compute_cost(c)
        if cost < best_cost:
            best_cost = cost
            best_c = c[:]
    return best_c

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    global_best_A = _generate_initial_candidate()
    global_best_cost = _compute_cost(global_best_A)
    
    # Multiple restarts
    while time.time() < deadline - 10.0:
        current_A = global_best_A[:]
        current_cost = global_best_cost
        
        # Sometimes start from a different heuristic
        if random.random() < 0.3:
            current_A = _generate_initial_candidate()
            current_cost = _compute_cost(current_A)
        
        T = 1000.0
        T_min = 0.1
        alpha = 0.999
        no_improve_count = 0
        
        while time.time() < deadline - 5.0:
            new_A = current_A[:]
            move_type = random.random()
            
            if move_type < 0.5:
                i, j = random.sample(range(20), 2)
                new_A[i], new_A[j] = new_A[j], new_A[i]
            elif move_type < 0.8:
                i = random.randint(0, 19)
                delta = random.randint(-20, 20)
                new_val = current_A[i] + delta
                if 1 <= new_val <= 10000 and new_val not in new_A:
                    new_A[i] = new_val
                else:
                    # Fallback to random small change
                    new_val = random.randint(1, 10000)
                    while new_val in new_A:
                        new_val = random.randint(1, 10000)
                    new_A[i] = new_val
            else:
                # Larger jump
                i = random.randint(0, 19)
                new_val = random.randint(1, 10000)
                while new_val in new_A:
                    new_val = random.randint(1, 10000)
                new_A[i] = new_val

            new_cost = _compute_cost(new_A)
            
            if new_cost < current_cost:
                current_A = new_A
                current_cost = new_cost
                no_improve_count = 0
                if current_cost < global_best_cost:
                    global_best_cost = current_cost
                    global_best_A = current_A[:]
            else:
                delta = new_cost - current_cost
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_A = new_A
                        current_cost = new_cost
                no_improve_count += 1
            
            T *= alpha
            if T < T_min:
                T = T_min
                
            # Restart SA if stuck
            if no_improve_count > 1000:
                break
                
    return global_best_A
