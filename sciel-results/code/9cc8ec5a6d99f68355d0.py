import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    num_masks = 1 << n  # 4096
    
    # Precompute comparability matrix for speed
    # comp[a][b] = 1 if a and b are comparable (one is strict subset of other)
    comp = [0] * (num_masks * num_masks)
    for a in range(num_masks):
        for b in range(num_masks):
            if a != b and ((a & b) == a or (a & b) == b):
                comp[a * num_masks + b] = 1
    
    def is_comparable(a, b):
        return comp[a * num_masks + b]

    def calc_score(masks):
        s = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if comp[masks[i] * num_masks + masks[j]]:
                    s += 1
        return s

    # 1. Greedy Initialization
    # Start with empty set and full set
    current_masks = [0, full_mask]
    used_masks = {0, full_mask}
    
    # Add remaining m-2 sets greedily
    for _ in range(m - 2):
        best_mask = -1
        best_score = -1
        # Iterate through all possible masks
        for mask in range(1, full_mask):
            if mask in used_masks:
                continue
            score = 0
            for cm in current_masks:
                if comp[mask * num_masks + cm]:
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        
        if best_mask != -1:
            current_masks.append(best_mask)
            used_masks.add(best_mask)
        else:
            mask = random.randint(0, full_mask)
            while mask in used_masks:
                mask = random.randint(0, full_mask)
            current_masks.append(mask)
            used_masks.add(mask)

    best_score = calc_score(current_masks)
    best_solution = list(current_masks)
    
    # 2. Simulated Annealing
    random.seed(42)
    
    # Parameters
    T = 10.0
    cooling_rate = 0.9998
    min_T = 0.01
    
    restarts = 0
    max_restarts = 8 # Increased restarts to leverage faster comparability checks
    
    while time.time() < deadline and restarts < max_restarts:
        current_masks = list(best_solution)
        used_masks = set(current_masks)
        current_score = best_score
        
        T = 10.0
        no_improve_steps = 0
        max_no_improve = 10000
        
        while time.time() < deadline and T > min_T:
            # Pick a random index to modify
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]
            
            # Generate new mask
            if random.random() < 0.5:
                # Flip a bit
                if old_mask == 0:
                    # Can only add
                    bit = random.randint(0, n - 1)
                    new_mask = old_mask | (1 << bit)
                elif old_mask == full_mask:
                    # Can only remove
                    bit = random.randint(0, n - 1)
                    new_mask = old_mask ^ (1 << bit)
                else:
                    # Randomly add or remove a bit
                    if random.random() < 0.5:
                        # Remove
                        bits = [i for i in range(n) if (old_mask >> i) & 1]
                        bit = random.choice(bits)
                        new_mask = old_mask ^ (1 << bit)
                    else:
                        # Add
                        bits = [i for i in range(n) if not ((old_mask >> i) & 1)]
                        bit = random.choice(bits)
                        new_mask = old_mask | (1 << bit)
            else:
                # Random mask
                new_mask = random.randint(0, full_mask)
            
            if new_mask == old_mask or new_mask in used_masks:
                continue

            # Calculate delta using precomputed matrix
            delta = 0
            for j in range(m):
                if j == idx:
                    continue
                old_comp = comp[old_mask * num_masks + current_masks[j]]
                new_comp = comp[new_mask * num_masks + current_masks[j]]
                delta += new_comp - old_comp
            
            # Acceptance probability
            if delta > 0:
                # Accept
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                current_score += delta
                no_improve_steps = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_solution = list(current_masks)
            else:
                # Reject or accept with probability exp(delta/T)
                prob = math.exp(delta / T)
                if random.random() < prob:
                    used_masks.remove(old_mask)
                    used_masks.add(new_mask)
                    current_masks[idx] = new_mask
                    current_score += delta
                    no_improve_steps += 1
                else:
                    no_improve_steps += 1
            
            T *= cooling_rate
            
            if no_improve_steps > max_no_improve:
                break

        restarts += 1

    # Convert to list of lists
    result = []
    for mask in best_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result
