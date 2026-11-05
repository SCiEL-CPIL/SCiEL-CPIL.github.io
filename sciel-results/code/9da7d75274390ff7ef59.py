import time
import random
import math

N = 256

def elem_to_bit(e):
    return 1 << (e - 1)

# Precompute 4-AP conflicts and 3-AP gains
four_conflicts = [[] for _ in range(N + 1)]
three_gains = [[] for _ in range(N + 1)]

for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
        four_conflicts[a].append(elem_to_bit(b) | elem_to_bit(c) | elem_to_bit(e))
        four_conflicts[b].append(elem_to_bit(a) | elem_to_bit(c) | elem_to_bit(e))
        four_conflicts[c].append(elem_to_bit(a) | elem_to_bit(b) | elem_to_bit(e))
        four_conflicts[e].append(elem_to_bit(a) | elem_to_bit(b) | elem_to_bit(c))

for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        three_gains[a].append(elem_to_bit(b) | elem_to_bit(c))
        three_gains[b].append(elem_to_bit(a) | elem_to_bit(c))
        three_gains[c].append(elem_to_bit(a) | elem_to_bit(b))

def is_4ap_free_addition_mask(A_mask, x):
    for m in four_conflicts[x]:
        if (A_mask & m) == m:
            return False
    return True

def delta_3aps_for_addition_mask(A_mask, x):
    count = 0
    for m in three_gains[x]:
        if (A_mask & m) == m:
            count += 1
    return count

def count_3aps_mask(A_mask):
    total = 0
    for i in range(1, N + 1):
        if (A_mask >> (i - 1)) & 1:
            total += delta_3aps_for_addition_mask(A_mask, i)
    return total // 3

def greedy_initial_mask(seed_val=0):
    rng = random.Random(seed_val)
    A_mask = 0
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    for x in candidates:
        if is_4ap_free_addition_mask(A_mask, x):
            A_mask |= elem_to_bit(x)
    return A_mask

def simulated_annealing_mask(A_mask_init, deadline, seed_val=0):
    rng = random.Random(seed_val)
    A_mask = A_mask_init
    current_score = count_3aps_mask(A_mask)
    best_mask = A_mask
    best_score = current_score
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.1
    cooling = 0.9995
    
    T = T_start
    
    while time.time() < deadline and T > T_end:
        # Generate a random move
        move_type = rng.random()
        new_mask = A_mask
        new_score = current_score
        
        if move_type < 0.3:
            # Try to add an element
            x = rng.randint(1, N)
            if not ((A_mask >> (x - 1)) & 1):
                if is_4ap_free_addition_mask(A_mask, x):
                    delta = delta_3aps_for_addition_mask(A_mask, x)
                    new_mask = A_mask | elem_to_bit(x)
                    new_score = current_score + delta
        elif move_type < 0.6:
            # Try to remove an element
            x = rng.randint(1, N)
            if (A_mask >> (x - 1)) & 1:
                delta = delta_3aps_for_addition_mask(A_mask, x)
                new_mask = A_mask & ~elem_to_bit(x)
                new_score = current_score - delta
        else:
            # Try a swap
            x = rng.randint(1, N)
            if (A_mask >> (x - 1)) & 1:
                A_temp = A_mask & ~elem_to_bit(x)
                y = rng.randint(1, N)
                if y != x and not ((A_temp >> (y - 1)) & 1):
                    if is_4ap_free_addition_mask(A_temp, y):
                        loss = delta_3aps_for_addition_mask(A_mask, x)
                        gain = delta_3aps_for_addition_mask(A_temp, y)
                        new_mask = A_temp | elem_to_bit(y)
                        new_score = current_score - loss + gain
        
        # Accept or reject
        delta = new_score - current_score
        if delta >= 0 or rng.random() < math.exp(delta / T):
            A_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = A_mask
        
        T *= cooling
    
    return best_mask

def mask_to_list(A_mask):
    return [i for i in range(1, N + 1) if (A_mask >> (i - 1)) & 1]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_mask = 0
    best_score = -1
    
    # Run multiple SA restarts with different seeds
    for seed in range(10):
        if time.time() > deadline - 5:
            break
        
        init_mask = greedy_initial_mask(seed)
        sa_mask = simulated_annealing_mask(init_mask, deadline, seed)
        sa_score = count_3aps_mask(sa_mask)
        
        if sa_score > best_score:
            best_score = sa_score
            best_mask = sa_mask
    
    if best_mask == 0:
        best_mask = greedy_initial_mask(0)
    
    return mask_to_list(best_mask)
