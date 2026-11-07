import time
import random

def construct():
    n = 12
    m = 40
    # Use a safety margin well under the 90s wall-clock limit
    deadline = time.time() + 82.0
    
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]
    
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b

    def calculate_score(masks):
        score = 0
        for i in range(m):
            for j in range(i+1, m):
                if is_comparable(masks[i], masks[j]):
                    score += 1
        return score

    def score_delta(swap_idx, new_mask, masks):
        old_mask = masks[swap_idx]
        delta = 0
        for j in range(m):
            if j == swap_idx: continue
            other = masks[j]
            if is_comparable(new_mask, other):
                delta += 1
            if is_comparable(old_mask, other):
                delta -= 1
        return delta

    best_score = -1
    best_masks = None
    
    # We will run a loop of SA restarts
    # Each restart runs for a fixed number of steps or until time slice expires
    
    # To ensure we explore well, we mix mutation and random jumps
    
    for restart in range(100): # Max 100 restarts
        if time.time() > deadline:
            break
            
        # Initialization
        # Try a few init strategies
        if restart % 3 == 0:
             # Random
             masks = list(random.sample(range(1 << n), m))
        elif restart % 3 == 1:
             # Greedy-ish: Start with a chain, then add random
             # Chain of 12: 0, 1, 3, 7...
             # Actually, just random is fine, SA will fix it
             masks = list(random.sample(range(1 << n), m))
        else:
             # Clustered: Pick a random base, generate neighbors
             base = random.randint(0, (1 << n) - 1)
             candidates = set()
             candidates.add(base)
             # Generate 200 random masks near base
             for _ in range(200):
                 b = base
                 for _ in range(random.randint(1, 3)):
                     bit = random.randint(0, n-1)
                     b ^= (1 << bit)
                 candidates.add(b)
             # Add some fully random
             for _ in range(50):
                 candidates.add(random.randint(0, (1 << n) - 1))
             
             # Select 40 distinct
             masks_list = list(candidates)
             random.shuffle(masks_list)
             masks = masks_list[:m]
             # Ensure distinct
             # The set logic above ensures distinct in candidates, 
             # but if candidates < 40, we need to fill
             if len(masks) < m:
                 existing = set(masks)
                 while len(masks) < m:
                     nm = random.randint(0, (1 << n) - 1)
                     if nm not in existing:
                         masks.append(nm)
                         existing.add(nm)

        current_score = calculate_score(masks)
        
        # Simulated Annealing
        T = 10.0
        min_T = 0.05
        cooling = 0.999
        no_improve_count = 0
        max_no_improve = 5000
        
        local_best_score = current_score
        local_best_masks = masks[:]
        
        steps_in_run = 0
        max_steps_per_run = 20000 # Cap steps per restart to allow restarts
        
        while steps_in_run < max_steps_per_run and time.time() < deadline:
            steps_in_run += 1
            no_improve_count += 1
            
            # Generate Neighbor
            i = random.randint(0, m - 1)
            old_mask = masks[i]
            
            # 80% Mutation, 20% Random
            if random.random() < 0.8:
                new_mask = old_mask
                # Flip 1-3 bits
                num_flips = random.randint(1, 3)
                for _ in range(num_flips):
                    bit = random.randint(0, n - 1)
                    new_mask ^= (1 << bit)
            else:
                new_mask = random.randint(0, (1 << n) - 1)
            
            if new_mask == old_mask:
                continue
                
            # Check if new_mask is already in masks (excluding i)
            # Since we are swapping, if new_mask is at j != i, we have a duplicate issue
            # We must ensure the resulting set has distinct elements.
            # If new_mask is in masks[k] for k != i, we cannot just swap.
            # We would need to swap i with k? Or reject?
            # Rejecting is simpler.
            duplicate = False
            for k in range(m):
                if k != i and masks[k] == new_mask:
                    duplicate = True
                    break
            
            if duplicate:
                continue
            
            delta = score_delta(i, new_mask, masks)
            
            # Acceptance
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                masks[i] = new_mask
                current_score += delta
                
                if current_score > local_best_score:
                    local_best_score = current_score
                    local_best_masks = masks[:]
                    no_improve_count = 0
                    
            T *= cooling
            
            if no_improve_count > max_no_improve:
                break
        
        if local_best_score > best_score:
            best_score = local_best_score
            best_masks = local_best_masks[:]
            
    if best_masks is None:
        best_masks = list(range(m))
        
    return [mask_to_list(mask) for mask in best_masks]
