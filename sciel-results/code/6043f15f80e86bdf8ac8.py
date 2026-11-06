import time
import random
import math
import numpy as np

def construct():
    t0 = time.time()
    # Safety margin: 90s limit, aim to finish well before
    DEADLINE = t0 + 75.0
    
    N = 256
    
    # Precompute conflict masks
    # conflict_mask[i] is a bitmask of j such that (i+1, j+1) is a conflicting pair
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]
    
    def get_size(mask):
        return bin(mask).count('1')
    
    def can_add(idx, mask):
        return (conflict_mask[idx] & mask) == 0

    def valid_set(mask):
        indices = [i for i in range(N) if (mask >> i) & 1]
        for ii in range(len(indices)):
            for jj in range(ii+1, len(indices)):
                a = indices[ii]
                b = indices[jj]
                if (conflict_mask[a] >> b) & 1:
                    return False
        return True

    best_mask = 0
    best_size = 0
    
    # Strategy 1: Greedy construction optimized for "Odds + Safe Evens"
    
    # 1a. Start with all odds
    mask_odds = 0
    for i in odds_indices:
        mask_odds |= (1 << i)
    
    # 1b. Identify evens and their conflicts with odds
    even_conflicts_with_odds = {}
    for i in evens_indices:
        conflicts = conflict_mask[i] & mask_odds
        count = bin(conflicts).count('1')
        even_conflicts_with_odds[i] = (count, conflicts)
    
    # Sort evens by conflict count, then by index
    evens_sorted = sorted(evens_indices, key=lambda i: (even_conflicts_with_odds[i][0], i))
    
    # Greedy add evens
    current_mask = mask_odds
    for i in evens_sorted:
        if can_add(i, current_mask):
            current_mask |= (1 << i)
    
    # Second pass: Try to add evens that conflict with 1 odd by removing that odd
    remaining_evens = [i for i in evens_indices if not ((current_mask >> i) & 1)]
    
    def conflict_count_with_mask(idx, m):
        return bin(conflict_mask[idx] & m).count('1')
    
    remaining_evens.sort(key=lambda i: conflict_count_with_mask(i, current_mask))
    
    for i in remaining_evens:
        conflicts = conflict_mask[i] & current_mask
        count = bin(conflicts).count('1')
        if count == 0:
            current_mask |= (1 << i)
        elif count == 1:
            c_idx = (conflicts).bit_length() - 1
            if (c_idx + 1) % 2 == 1:
                trial_mask = current_mask & ~(1 << c_idx)
                if can_add(i, trial_mask):
                    current_mask = trial_mask | (1 << i)
    
    sz = get_size(current_mask)
    if sz > best_size:
        best_mask = current_mask
        best_size = sz

    # Strategy 2: Local Search (Simulated Annealing)
    
    def neighborhood_move(mask, size):
        indices_in = [i for i in range(N) if (mask >> i) & 1]
        indices_out = [i for i in range(N) if not ((mask >> i) & 1)]
        
        if not indices_in or not indices_out:
            return mask, size, False
        
        # Try 1-remove 2-add (net +1)
        for _ in range(5):
            to_remove = random.choice(indices_in)
            trial = mask & ~(1 << to_remove)
            
            candidates = [i for i in indices_out if can_add(i, trial)]
            
            if len(candidates) >= 2:
                for _ in range(10):
                    e1_idx = random.randint(0, len(candidates) - 1)
                    e2_idx = random.randint(0, len(candidates) - 1)
                    if e1_idx == e2_idx:
                        continue
                    e1 = candidates[e1_idx]
                    e2 = candidates[e2_idx]
                    if not ((conflict_mask[e1] >> e2) & 1):
                        new_mask = trial | (1 << e1) | (1 << e2)
                        return new_mask, size + 1, True
        
        # Try 1-remove 1-add (net 0)
        for _ in range(5):
            to_remove = random.choice(indices_in)
            trial = mask & ~(1 << to_remove)
            
            candidates = [i for i in indices_out if can_add(i, trial)]
            
            if candidates:
                e1 = random.choice(candidates)
                new_mask = trial | (1 << e1)
                return new_mask, size, True
        
        return mask, size, False

    current_mask = best_mask
    current_size = best_size
    
    temperature = 1.0
    cooling = 0.999
    min_temp = 0.01
    
    # Run SA until time limit
    while time.time() < DEADLINE - 5:
        new_mask, new_size, made_move = neighborhood_move(current_mask, current_size)
        
        if made_move:
            if new_size >= current_size:
                current_mask = new_mask
                current_size = new_size
                if new_size > best_size:
                    best_mask = new_mask
                    best_size = new_size
            else:
                delta = current_size - new_size
                if random.random() < math.exp(-delta / max(temperature, 1e-10)):
                    current_mask = new_mask
                    current_size = new_size
        
        temperature = max(temperature * cooling, min_temp)
        
        # Reheat occasionally
        if random.random() < 0.01:
            temperature = 0.5

    # Final polish: try to add any node that fits
    for i in range(N):
        if not ((best_mask >> i) & 1) and can_add(i, best_mask):
            best_mask |= (1 << i)
    
    if not valid_set(best_mask):
        best_mask = 0
        for i in odds_indices:
            best_mask |= (1 << i)
    
    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
    return result
