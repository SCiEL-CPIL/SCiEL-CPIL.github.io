import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 58.0

    # Precompute APs
    ap_list = []
    for d in range(1, N + 1):
        for start in range(0, N - d + 1):
            terms = list(range(start, N, d))
            if len(terms) >= 2:
                ap_list.append(terms)
    
    num_aps = len(ap_list)
    
    # Dense membership matrix
    M = np.zeros((num_aps, N), dtype=np.int8)
    for ap_idx, terms in enumerate(ap_list):
        for pos in terms:
            M[ap_idx, pos] = 1

    def compute_sums(signs):
        return M @ signs.astype(np.int32)

    def get_max_abs(sums):
        return int(np.max(np.abs(sums)))

    def greedy_init():
        signs = np.zeros(N, dtype=np.int8)
        ap_sums = np.zeros(num_aps, dtype=np.int32)
        
        for i in range(N):
            best_sign = 1
            best_max = N + 1
            best_sums = None
            
            for s in [1, -1]:
                affected = M[:, i]
                new_sums = ap_sums + s * affected
                m = get_max_abs(new_sums)
                if m < best_max:
                    best_max = m
                    best_sign = s
                    best_sums = new_sums
            
            signs[i] = best_sign
            ap_sums = best_sums
        return signs, ap_sums

    # Initial strong solution
    init_signs, init_sums = greedy_init()
    best_val = get_max_abs(init_sums)
    best_signs = init_signs.copy()

    # Population setup
    pop_size = 20
    population = [best_signs.copy() for _ in range(pop_size)]
    
    # Add diversity to initial population
    for i in range(1, pop_size):
        rand_signs = np.random.choice([-1, 1], N).astype(np.int8)
        # Crossover with best
        mask = np.random.rand(N) < 0.5
        population[i] = np.where(mask, best_signs, rand_signs).astype(np.int8)
        
    # Evaluate initial pop
    fitnesses = [get_max_abs(compute_sums(ind)) for ind in population]
    
    # Track global best
    global_best_val = min(fitnesses)
    global_best_signs = population[fitnesses.index(global_best_val)].copy()

    def repair(signs, time_limit):
        nonlocal global_best_val, global_best_signs
        
        signs = signs.copy()
        sums = compute_sums(signs)
        current_max = get_max_abs(sums)
        
        local_best_val = current_max
        local_best_signs = signs.copy()
        
        while time.time() < time_limit:
            max_idx = np.argmax(np.abs(sums))
            max_sum = sums[max_idx]
            
            if max_sum == 0:
                break
                
            terms = ap_list[max_idx]
            target_sign = 1 if max_sum > 0 else -1
            
            best_flip_idx = -1
            best_new_max = current_max
            best_new_sums = None
            
            for pos in terms:
                if signs[pos] == target_sign:
                    delta = -2 * signs[pos]
                    new_sums = sums + delta * M[:, pos]
                    new_max = get_max_abs(new_sums)
                    
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_flip_idx = pos
                        best_new_sums = new_sums
            
            if best_flip_idx != -1 and best_new_max < current_max:
                signs[best_flip_idx] = -signs[best_flip_idx]
                sums = best_new_sums
                current_max = best_new_max
                
                if current_max < local_best_val:
                    local_best_val = current_max
                    local_best_signs = signs.copy()
            else:
                pos = random.randint(0, N - 1)
                delta = -2 * signs[pos]
                new_sums = sums + delta * M[:, pos]
                new_max = get_max_abs(new_sums)
                
                if new_max < current_max:
                    signs[pos] = -signs[pos]
                    sums = new_sums
                    current_max = new_max
                    if current_max < local_best_val:
                        local_best_val = current_max
                        local_best_signs = signs.copy()
                elif new_max == current_max and random.random() < 0.1:
                    signs[pos] = -signs[pos]
                    sums = new_sums
                elif new_max > current_max and random.random() < 0.005:
                    signs[pos] = -signs[pos]
                    sums = new_sums
                    current_max = new_max
                else:
                    break
                    
        if local_best_val < global_best_val:
            global_best_val = local_best_val
            global_best_signs = local_best_signs.copy()
            
        return local_best_signs, local_best_val

    # Main GA Loop
    while time.time() < deadline - 5.0:
        # Select parents
        p1_idx = random.randint(0, pop_size - 1)
        p2_idx = random.randint(0, pop_size - 1)
        
        # Crossover
        child = np.zeros(N, dtype=np.int8)
        for i in range(N):
            if random.random() < 0.5:
                child[i] = population[p1_idx][i]
            else:
                child[i] = population[p2_idx][i]
                
        # Mutation
        if random.random() < 0.3:
            mut_idx = random.randint(0, N - 1)
            child[mut_idx] = -child[mut_idx]
            
        # Repair
        child_deadline = time.time() + 0.4
        child, child_max = repair(child, child_deadline)
        
        # Update population
        worst_idx = np.argmax(fitnesses)
        if child_max < fitnesses[worst_idx]:
            population[worst_idx] = child
            fitnesses[worst_idx] = child_max

    return global_best_signs.tolist()
