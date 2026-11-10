import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    total_masks = 1 << n  # 4096
    
    # Precompute comparability matrix: comp[a][b] = 1 if a and b are comparable (strict subset in one direction)
    # Using a flat list for speed: comp[a * total_masks + b]
    comp = [0] * (total_masks * total_masks)
    for a in range(total_masks):
        base_a = a * total_masks
        for b in range(total_masks):
            if a == b:
                comp[base_a + b] = 0
            elif (a & b) == a or (a & b) == b:
                comp[base_a + b] = 1
            else:
                comp[base_a + b] = 0
    
    def calc_score(masks):
        s = 0
        for i in range(m):
            base_i = masks[i] * total_masks
            for j in range(i + 1, m):
                s += comp[base_i + masks[j]]
        return s
    
    def greedy_chain_init(seed_val):
        """Initialize with a chain structure, then greedily add sets that maximize comparability."""
        rng = random.Random(seed_val)
        elem_order = list(range(n))
        rng.shuffle(elem_order)
        
        chain = [0]
        cur = 0
        for e in elem_order:
            cur |= (1 << e)
            chain.append(cur)
        
        current_masks = list(chain)
        used_masks = set(current_masks)
        
        # Precompute scores for each candidate mask against current set
        for _ in range(m - len(current_masks)):
            best_mask = -1
            best_score = -1
            for mask in range(total_masks):
                if mask in used_masks:
                    continue
                score = 0
                base_mask = mask * total_masks
                for cm in current_masks:
                    score += comp[base_mask + cm]
                if score > best_score:
                    best_score = score
                    best_mask = mask
            current_masks.append(best_mask)
            used_masks.add(best_mask)
        
        return current_masks
    
    random.seed(42)
    
    best_global_score = -1
    best_global_solution = None
    
    num_restarts = 12
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        current_masks = greedy_chain_init(42 + r)
        used_masks = set(current_masks)
        
        cur_score = calc_score(current_masks)
        
        if cur_score > best_global_score:
            best_global_score = cur_score
            best_global_solution = list(current_masks)
        
        # Local Search with improved delta calculation
        stuck = 0
        max_steps_per_restart = 50000
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline - 3.0:
                break
            
            # Pick a random index to modify
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]
            
            # Generate a new mask
            new_mask = old_mask
            op = random.random()
            if op < 0.30:
                # Add bit
                zeros = [i for i in range(n) if not ((old_mask >> i) & 1)]
                if zeros:
                    bit = random.choice(zeros)
                    new_mask = old_mask | (1 << bit)
            elif op < 0.60:
                # Remove bit
                ones = [i for i in range(n) if (old_mask >> i) & 1]
                if ones:
                    bit = random.choice(ones)
                    new_mask = old_mask ^ (1 << bit)
            elif op < 0.80:
                # Swap bit
                ones = [i for i in range(n) if (old_mask >> i) & 1]
                zeros = [i for i in range(n) if not ((old_mask >> i) & 1)]
                if ones and zeros:
                    bo = random.choice(ones)
                    bz = random.choice(zeros)
                    new_mask = (old_mask ^ (1 << bo)) | (1 << bz)
            else:
                # Random jump
                new_mask = random.randint(0, full_mask)
                
            if new_mask == old_mask:
                continue
                
            if new_mask in used_masks:
                continue
                
            # Calculate Delta efficiently using precomputed comparability
            delta = 0
            base_old = old_mask * total_masks
            base_new = new_mask * total_masks
            for j in range(m):
                if j == idx:
                    continue
                mask_j = current_masks[j]
                delta += comp[base_new + mask_j] - comp[base_old + mask_j]
                
            if delta > 0:
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                cur_score += delta
                stuck = 0
                if cur_score > best_global_score:
                    best_global_score = cur_score
                    best_global_solution = list(current_masks)
            else:
                stuck += 1
                if stuck > 80:
                    # Random kick: replace 2-3 random sets
                    kick_size = random.randint(2, 3)
                    for _ in range(kick_size):
                        idx2 = random.randint(0, m - 1)
                        old_mask2 = current_masks[idx2]
                        while True:
                            new_mask2 = random.randint(0, full_mask)
                            if new_mask2 not in used_masks:
                                break
                        used_masks.remove(old_mask2)
                        used_masks.add(new_mask2)
                        current_masks[idx2] = new_mask2
                    cur_score = calc_score(current_masks)
                    stuck = 0
        
        # Final polish pass: try all single-bit moves for each position
        if time.time() > deadline - 2.0:
            continue
        
        polished = True
        while polished:
            if time.time() > deadline - 1.0:
                break
            polished = False
            for idx in range(m):
                old_mask = current_masks[idx]
                for bit in range(n):
                    # Try adding bit
                    if not ((old_mask >> bit) & 1):
                        new_mask = old_mask | (1 << bit)
                        if new_mask not in used_masks:
                            delta = 0
                            base_old = old_mask * total_masks
                            base_new = new_mask * total_masks
                            for j in range(m):
                                if j == idx:
                                    continue
                                delta += comp[base_new + current_masks[j]] - comp[base_old + current_masks[j]]
                            if delta > 0:
                                used_masks.remove(old_mask)
                                used_masks.add(new_mask)
                                current_masks[idx] = new_mask
                                cur_score += delta
                                polished = True
                                if cur_score > best_global_score:
                                    best_global_score = cur_score
                                    best_global_solution = list(current_masks)
                    # Try removing bit
                    if (old_mask >> bit) & 1:
                        new_mask = old_mask ^ (1 << bit)
                        if new_mask not in used_masks:
                            delta = 0
                            base_old = old_mask * total_masks
                            base_new = new_mask * total_masks
                            for j in range(m):
                                if j == idx:
                                    continue
                                delta += comp[base_new + current_masks[j]] - comp[base_old + current_masks[j]]
                            if delta > 0:
                                used_masks.remove(old_mask)
                                used_masks.add(new_mask)
                                current_masks[idx] = new_mask
                                cur_score += delta
                                polished = True
                                if cur_score > best_global_score:
                                    best_global_score = cur_score
                                    best_global_solution = list(current_masks)
    
    # Convert best solution to list of lists
    result = []
    for mask in best_global_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
    
    return result
