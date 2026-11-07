import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Helper to check comparability
    def is_comp(a, b):
        return (a & b) == a or (a & b) == b

    # Generate an initial chain of length 13
    initial_chain = [(1 << i) - 1 for i in range(n + 1)]
    
    family = initial_chain[:]
    in_family = set(family)
    
    # Calculate initial score
    current_score = len(family) * (len(family) - 1) // 2
    
    # Greedily add remaining sets
    for _ in range(m - len(family)):
        best_mask = -1
        best_gain = -1
        
        for mask in range(total_masks):
            if mask in in_family:
                continue
            
            gain = 0
            for f in family:
                if is_comp(mask, f):
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
                
        if best_mask != -1:
            family.append(best_mask)
            in_family.add(best_mask)
            current_score += best_gain
        else:
            break

    best_family = family[:]
    best_score = current_score

    # Simulated Annealing to improve
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        i = random.randint(0, m - 1)
        old_mask = family[i]
        
        new_mask = random.randint(0, total_masks - 1)
        if new_mask in in_family:
            continue

        # Calculate delta
        delta = 0
        for j in range(m):
            if i == j:
                continue
            other = family[j]
            if is_comp(old_mask, other) and not is_comp(new_mask, other):
                delta -= 1
            elif not is_comp(old_mask, other) and is_comp(new_mask, other):
                delta += 1
        
        if delta >= 0 or random.random() < pow(2.71828, -delta / T):
            family[i] = new_mask
            in_family.remove(old_mask)
            in_family.add(new_mask)
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_family = family[:]
        
        T *= cooling
        if T < 0.1:
            T = 0.1

    # Convert best_family to list of lists of ints
    result = []
    for mask in best_family:
        result.append([b for b in range(n) if (mask >> b) & 1])
        
    return result
