import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    num_masks = 1 << n  # 4096
    
    # Precompute comparability matrix: comp[a][b] = 1 if a is strict subset of b or vice versa
    comp = [0] * (num_masks * num_masks)
    for a in range(num_masks):
        for b in range(num_masks):
            if a != b and ((a & b) == a or (a & b) == b):
                comp[a * num_masks + b] = 1
    
    def calc_score(masks):
        s = 0
        for i in range(m):
            for j in range(i + 1, m):
                if comp[masks[i] * num_masks + masks[j]]:
                    s += 1
        return s

    def greedy_init(seed_val):
        """Greedy initialization with a random tie-breaker for diversity."""
        rng = random.Random(seed_val)
        current_masks = [0, full_mask]
        used_masks = {0, full_mask}
        
        for _ in range(m - 2):
            best_mask = -1
            best_score = -1
            best_tie = 0
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
                    best_tie = 0
                elif score == best_score:
                    best_tie += 1
                    if rng.random() < 0.5:
                        best_mask = mask
        
        if best_mask != -1:
            current_masks.append(best_mask)
            used_masks.add(best_mask)
        else:
            break
        return current_masks

    # 1. Generate multiple diverse greedy initial solutions
    random.seed(42)
    best_score = -1
    best_solution = None
    
    num_initials = 5
    for i in range(num_initials):
        init = greedy_init(i * 1000 + 42)
        if len(init) == m:
            sc = calc_score(init)
            if sc > best_score:
                best_score = sc
                best_solution = list(init)
    
    # If greedy didn't fill all slots, fill remaining randomly
    if len(best_solution) < m:
        used = set(best_solution)
        while len(best_solution) < m:
            cm = random.randint(0, num_masks - 1)
            if cm not in used:
                best_solution.append(cm)
                used.add(cm)
        best_score = calc_score(best_solution)
    
    # 2. Simulated Annealing with multiple restarts
    max_restarts = 30
    restarts = 0
    
    while time.time() < deadline - 5.0 and restarts < max_restarts:
        if restarts == 0:
            current_masks = list(best_solution)
        else:
            # Start from best solution with random perturbation
            current_masks = list(best_solution)
            for _ in range(random.randint(5, 15)):
                idx = random.randint(0, m - 1)
                new_mask = random.randint(0, num_masks - 1)
                if new_mask not in current_masks:
                    current_masks[idx] = new_mask
            current_masks = list(set(current_masks))
            # Ensure we have exactly m distinct masks
            used = set(current_masks)
            while len(current_masks) < m:
                cm = random.randint(0, num_masks - 1)
                if cm not in used:
                    current_masks.append(cm)
                    used.add(cm)
            current_masks = current_masks[:m]
        
        used_masks = set(current_masks)
        current_score = calc_score(current_masks)
        
        T = 3.0
        cooling_rate = 0.99997
        min_T = 0.0005
        no_improve_steps = 0
        max_no_improve = 8000
        
        while time.time() < deadline - 5.0 and T > min_T:
            # Pick a random index to modify
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]
            
            # Generate new mask
            r = random.random()
            if r < 0.5:
                # Flip one bit
                if old_mask == 0:
                    bit = random.randint(0, n - 1)
                    new_mask = old_mask | (1 << bit)
                elif old_mask == full_mask:
                    bit = random.randint(0, n - 1)
                    new_mask = old_mask ^ (1 << bit)
                else:
                    if random.random() < 0.5:
                        bits = [i for i in range(n) if (old_mask >> i) & 1]
                        bit = random.choice(bits)
                        new_mask = old_mask ^ (1 << bit)
                    else:
                        bits = [i for i in range(n) if not ((old_mask >> i) & 1)]
                        bit = random.choice(bits)
                        new_mask = old_mask | (1 << bit)
            elif r < 0.7:
                # Flip two bits
                if old_mask == 0 or old_mask == full_mask:
                    new_mask = random.randint(0, num_masks - 1)
                else:
                    bits_on = [i for i in range(n) if (old_mask >> i) & 1]
                    bits_off = [i for i in range(n) if not ((old_mask >> i) & 1)]
                    if len(bits_on) >= 1 and len(bits_off) >= 1:
                        b1 = random.choice(bits_on)
                        b2 = random.choice(bits_off)
                        new_mask = old_mask ^ (1 << b1) ^ (1 << b2)
                    else:
                        new_mask = random.randint(0, num_masks - 1)
            else:
                # Random mask
                new_mask = random.randint(0, num_masks - 1)
            
            if new_mask == old_mask:
                continue
            
            if new_mask in used_masks:
                continue
            
            # Calculate delta efficiently
            delta = 0
            for j in range(m):
                if j == idx:
                    continue
                old_comp = comp[old_mask * num_masks + current_masks[j]]
                new_comp = comp[new_mask * num_masks + current_masks[j]]
                delta += new_comp - old_comp
            
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
                # Simulated annealing acceptance
                if T > min_T and T > 0:
                    p = min(1.0, (delta / T))
                    if random.random() < p:
                        # Accept bad move
                        used_masks.remove(old_mask)
                        used_masks.add(new_mask)
                        current_masks[idx] = new_mask
                        current_score += delta
                        no_improve_steps += 1
                    else:
                        no_improve_steps += 1
                else:
                    no_improve_steps += 1
            
            # Cool down
            T *= cooling_rate
            
            if no_improve_steps > max_no_improve:
                break
        
        restarts += 1
    
    # 3. Final polish: hill climbing with no bad moves
    current_masks = list(best_solution)
    used_masks = set(current_masks)
    current_score = calc_score(current_masks)
    
    improved = True
    polish_deadline = deadline - 1.0
    while improved and time.time() < polish_deadline:
        improved = False
        for idx in range(m):
            old_mask = current_masks[idx]
            best_new_mask = old_mask
            best_delta = 0
            for new_mask in range(num_masks):
                if new_mask == old_mask or new_mask in used_masks:
                    continue
                delta = 0
                for j in range(m):
                    if j == idx:
                        continue
                    old_comp = comp[old_mask * num_masks + current_masks[j]]
                    new_comp = comp[new_mask * num_masks + current_masks[j]]
                    delta += new_comp - old_comp
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask
            if best_delta > 0:
                used_masks.remove(old_mask)
                used_masks.add(best_new_mask)
                current_masks[idx] = best_new_mask
                current_score += best_delta
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_solution = list(current_masks)
    
    # Convert to list of lists
    result = []
    for mask in best_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
    
    return result
