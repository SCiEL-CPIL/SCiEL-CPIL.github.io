import random
import time
import math

# EVOLVE-BLOCK-START
def construct():
    MAX_N = 1000
    MAX_SUM = 2 * MAX_N
    
    def greedy_construct(order):
        counts = [0] * (MAX_SUM + 1)
        chosen = []
        chosen_set = set()
        for x in order:
            valid = True
            for y in chosen:
                s = x + y
                if counts[s] >= 2:
                    valid = False
                    break
            if valid:
                if counts[2 * x] >= 2:
                    valid = False
            
            if valid:
                for y in chosen:
                    counts[x + y] += 1
                counts[2 * x] += 1
                chosen.append(x)
                chosen_set.add(x)
        return chosen, chosen_set, counts

    # 1. Get a good initial solution
    candidates = list(range(1, MAX_N + 1))
    best_initial = []
    best_initial_size = 0
    best_initial_counts = []
    best_initial_set = set()
    
    for _ in range(20):
        random.shuffle(candidates)
        chosen, chosen_set, counts = greedy_construct(candidates)
        if len(chosen) > best_initial_size:
            best_initial_size = len(chosen)
            best_initial = chosen
            best_initial_counts = counts
            best_initial_set = chosen_set

    # 2. Simulated Annealing
    current_set = set(best_initial)
    current_counts = best_initial_counts[:]
    current_size = len(current_set)
    
    best_set = set(best_initial)
    best_size = current_size
    best_counts = best_initial_counts[:]
    
    # Helper to calculate delta V for adding x
    def delta_v_add(x, counts, S):
        dv = 0
        for y in S:
            s = x + y
            if counts[s] >= 2:
                dv += 1
        if counts[2 * x] >= 2:
            dv += 1
        return dv

    # Helper to calculate delta V for removing x
    def delta_v_remove(x, counts, S):
        dv = 0
        for y in S:
            s = x + y
            if counts[s] >= 3:
                dv -= 1
        return dv

    T = 100.0
    T_min = 0.01
    cooling = 0.999
    steps = 50000 
    start_time = time.time()
    
    excluded = [x for x in range(1, MAX_N + 1) if x not in current_set]
    
    for step in range(steps):
        if time.time() - start_time > 110:
            break
            
        # Pick a move: Add (50%), Remove (50%)
        # Swaps are effectively sequences of Remove+Add
        r = random.random()
        if r < 0.5 and excluded:
            # Add
            x = random.choice(excluded)
            dv = delta_v_add(x, current_counts, current_set)
            delta_cost = -1 + 100 * dv
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Accept
                for y in current_set:
                    current_counts[x + y] += 1
                current_counts[2 * x] += 1
                current_set.add(x)
                excluded.remove(x)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
                    best_counts = current_counts[:]
        else:
            # Remove
            if current_set:
                x = random.choice(list(current_set))
                dv = delta_v_remove(x, current_counts, current_set)
                delta_cost = 1 + 100 * dv
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    # Accept
                    for y in current_set:
                        current_counts[x + y] -= 1
                    current_counts[2 * x] -= 1
                    current_set.remove(x)
                    excluded.append(x)
                    current_size -= 1
        T *= cooling
        if T < T_min:
            T = T_min

    return list(best_set)
# EVOLVE-BLOCK-END
