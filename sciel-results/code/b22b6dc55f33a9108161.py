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
    
    # Precompute comparability matrix: comp[a][b] = 1 if a is strict subset of b or vice versa
    # Use a flat array for speed: comp[a * num_masks + b]
    comp = [0] * (num_masks * num_masks)
    for a in range(num_masks):
        for b in range(num_masks):
            if a != b and ((a & b) == a or (a & b) == b):
                comp[a * num_masks + b] = 1
    
    def calc_score(masks):
        s = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if comp[masks[i] * num_masks + masks[j]]:
                    s += 1
        return s
    
    def greedy_polish(masks, used_set):
        """Try to improve by swapping in better masks. Returns improved masks and score."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(m):
                old_mask = masks[i]
                best_swap = -1
                best_gain = 0
                old_base = old_mask * num_masks
                # Try all unused masks
                for new_mask in range(num_masks):
                    if new_mask in used_set or new_mask == old_mask:
                        continue
                    new_base = new_mask * num_masks
                    gain = 0
                    for j in range(m):
                        if j == i:
                            continue
                        cm = masks[j]
                        gain += comp[new_base + cm] - comp[old_base + cm]
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = new_mask
                if best_swap != -1:
                    used_set.remove(old_mask)
                    used_set.add(best_swap)
                    masks[i] = best_swap
                    improved = True
        return masks, calc_score(masks)
    
    # 1. Greedy Initialization with precomputed matrix
    current_masks = [0, full_mask]
    used_masks = {0, full_mask}
    
    for _ in range(m - 2):
        best_mask = -1
        best_score = -1
        for mask in range(num_masks):
            if mask in used_masks:
                continue
            score = 0
            base = mask * num_masks
            for cm in current_masks:
                if comp[base + cm]:
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        
        if best_mask != -1:
            current_masks.append(best_mask)
            used_masks.add(best_mask)
        else:
            break
    
    best_score = calc_score(current_masks)
    best_solution = list(current_masks)
    
    # Apply initial polish
    polished, polished_score = greedy_polish(list(best_solution), set(best_solution))
    if polished_score > best_score:
        best_score = polished_score
        best_solution = polished
    
    # 2. Simulated Annealing with precomputed comparability
    random.seed(42)
    
    restarts = 0
    max_restarts = 20
    
    while time.time() < deadline and restarts < max_restarts:
        current_masks = list(best_solution)
        used_masks = set(current_masks)
        current_score = best_score
        
        T = 5.0
        cooling_rate = 0.99995
        min_T = 0.001
        no_improve_steps = 0
        max_no_improve = 5000
        
        while time.time() < deadline and T > min_T:
            # Pick a random index to modify
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]
            
            # Generate new mask: mostly small perturbations
            r = random.random()
            if r < 0.6:
                # Flip one bit
                if old_mask == 0:
                    bit = random.randint(0, n - 1)
                    new_mask = old_mask | (1 << bit)
                elif old_mask == full_mask:
                    bit = random.randint(0, n - 1)
                    new_mask = old_mask ^ (1 << bit)
                else:
                    if random.random() < 0.5:
                        # Remove a bit
                        bits = [i for i in range(n) if (old_mask >> i) & 1]
                        bit = random.choice(bits)
                        new_mask = old_mask ^ (1 << bit)
                    else:
                        # Add a bit
                        bits = [i for i in range(n) if not ((old_mask >> i) & 1)]
                        bit = random.choice(bits)
                        new_mask = old_mask | (1 << bit)
            elif r < 0.8:
                # Flip two bits
                if old_mask == 0:
                    b1 = random.randint(0, n - 1)
                    b2 = random.randint(0, n - 1)
                    while b2 == b1:
                        b2 = random.randint(0, n - 1)
                    new_mask = (1 << b1) | (1 << b2)
                elif old_mask == full_mask:
                    b1 = random.randint(0, n - 1)
                    b2 = random.randint(0, n - 1)
                    while b2 == b1:
                        b2 = random.randint(0, n - 1)
                    new_mask = full_mask ^ (1 << b1) ^ (1 << b2)
                else:
                    all_bits = list(range(n))
                    random.shuffle(all_bits)
                    b1 = all_bits[0]
                    b2 = all_bits[1]
                    new_mask = old_mask ^ (1 << b1) ^ (1 << b2)
            elif r < 0.9:
                # Swap two elements in the set
                if len(current_masks) >= 2:
                    j = random.randint(0, m - 1)
                    if j != idx:
                        # Try swapping masks at positions idx and j
                        mask_i = current_masks[idx]
                        mask_j = current_masks[j]
                        # Calculate delta for swap
                        delta_swap = 0
                        base_i = mask_i * num_masks
                        base_j = mask_j * num_masks
                        for k in range(m):
                            if k == idx or k == j:
                                continue
                            cm = current_masks[k]
                            delta_swap += (comp[base_j + cm] + comp[base_i + cm]) - (comp[base_i + cm] + comp[base_j + cm])
                        # Actually swap just means we're replacing one mask with another
                        # Let's do a different move: pick a random unused mask
                        new_mask = random.randint(0, full_mask)
                        while new_mask in used_masks or new_mask == old_mask:
                            new_mask = random.randint(0, full_mask)
                    else:
                        new_mask = random.randint(0, full_mask)
                        while new_mask in used_masks or new_mask == old_mask:
                            new_mask = random.randint(0, full_mask)
                else:
                    new_mask = random.randint(0, full_mask)
                    while new_mask in used_masks or new_mask == old_mask:
                        new_mask = random.randint(0, full_mask)
            else:
                # Random mask
                new_mask = random.randint(0, full_mask)
            
            if new_mask == old_mask or new_mask in used_masks:
                continue

            # Calculate delta using precomputed matrix
            delta = 0
            old_base = old_mask * num_masks
            new_base = new_mask * num_masks
            for j in range(m):
                if j == idx:
                    continue
                cm = current_masks[j]
                old_c = comp[old_base + cm]
                new_c = comp[new_base + cm]
                delta += new_c - old_c
            
            # Acceptance
            if delta >= 0 or random.random() < math.exp(delta / T):
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                current_score += delta
                if delta > 0:
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                
                if current_score > best_score:
                    best_score = current_score
                    best_solution = list(current_masks)
            else:
                no_improve_steps += 1
            
            T *= cooling_rate
            
            if no_improve_steps > max_no_improve:
                break

        # Apply polish after each SA run
        polished, polished_score = greedy_polish(list(current_masks), set(current_masks))
        if polished_score > best_score:
            best_score = polished_score
            best_solution = polished

        restarts += 1

    # Final polish
    polished, polished_score = greedy_polish(list(best_solution), set(best_solution))
    if polished_score > best_score:
        best_score = polished_score
        best_solution = polished

    # Convert to list of lists
    result = []
    for mask in best_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result
