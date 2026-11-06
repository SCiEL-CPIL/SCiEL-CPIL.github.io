import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    # Precompute conflict_mask: conflict_mask[i] is a bitmask of indices j where i and j conflict
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)

    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]
    
    # Start with all odds
    current_mask = 0
    for i in odds_indices:
        current_mask |= (1 << i)
    
    best_mask = current_mask
    best_size = bin(current_mask).count('1')
    
    # Phase 1: Add evens with 0 conflicts
    # We do this in a greedy manner, updating mask as we go
    changed = True
    while changed and time.time() < DEADLINE - 50:
        changed = False
        for e in evens_indices:
            if time.time() > DEADLINE - 50:
                break
            if (current_mask >> e) & 1:
                continue
            if (conflict_mask[e] & current_mask) == 0:
                current_mask |= (1 << e)
                best_mask = current_mask
                best_size = bin(current_mask).count('1')
                changed = True
    
    # Phase 2: Simulated Annealing with improved moves
    T = 2.0
    cooling = 0.9995
    max_iters = 200000
    
    current_size = best_size
    # Precompute bit positions for faster iteration
    all_indices = list(range(N))
    
    for i_iter in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        r = random.random()
        if r < 0.7:
            # Single add move: add one element, remove its conflicts
            # Pick from evens first (80%), then odds (20%)
            if random.random() < 0.8:
                e_candidates = [idx for idx in evens_indices if not ((current_mask >> idx) & 1)]
                if not e_candidates:
                    continue
                e = random.choice(e_candidates)
            else:
                o_candidates = [idx for idx in odds_indices if not ((current_mask >> idx) & 1)]
                if not o_candidates:
                    continue
                e = random.choice(o_candidates)
            
            conflicts = conflict_mask[e] & current_mask
            num_conflicts = bin(conflicts).count('1')
            new_size = current_size - num_conflicts + 1
            
            if new_size > current_size:
                current_mask = (current_mask & ~conflicts) | (1 << e)
                current_size = new_size
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
            else:
                delta = new_size - current_size
                if delta == 0:
                    if random.random() < 0.5:
                        current_mask = (current_mask & ~conflicts) | (1 << e)
                else:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_mask = (current_mask & ~conflicts) | (1 << e)
                        current_size = new_size
        elif r < 0.9:
            # Multi-element exchange: remove k random elements, try to add one
            k = random.choice([1, 2, 3])
            bits = [i for i in range(N) if (current_mask >> i) & 1]
            if len(bits) < k:
                continue
            to_remove = random.sample(bits, k)
            
            # Build the mask without these k elements
            temp_mask = current_mask
            for rm in to_remove:
                temp_mask &= ~(1 << rm)
            
            # Try to add a random element not in temp_mask
            not_in_temp = [idx for idx in all_indices if not ((temp_mask >> idx) & 1)]
            if not not_in_temp:
                continue
            
            # Prefer evens
            not_in_evens = [idx for idx in not_in_temp if (idx + 1) % 2 == 0]
            if not_in_evens and random.random() < 0.7:
                e = random.choice(not_in_evens)
            else:
                e = random.choice(not_in_temp)
            
            conflicts = conflict_mask[e] & temp_mask
            num_conflicts = bin(conflicts).count('1')
            new_size = len(bits) - k - num_conflicts + 1
            
            if new_size > current_size:
                current_mask = (temp_mask & ~conflicts) | (1 << e)
                current_size = new_size
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
            else:
                delta = new_size - current_size
                if delta == 0:
                    if random.random() < 0.4:
                        current_mask = (temp_mask & ~conflicts) | (1 << e)
                else:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_mask = (temp_mask & ~conflicts) | (1 << e)
                        current_size = new_size
        else:
            # Random removal to escape local minima
            bits = [i for i in range(N) if (current_mask >> i) & 1]
            if bits:
                r = random.choice(bits)
                current_mask &= ~(1 << r)
                current_size -= 1
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Convert back to numbers
    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
    
    return result
