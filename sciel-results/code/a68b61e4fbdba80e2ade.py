import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    N = 32
    MAX_VAL = 512
    MAX_D = 255  # max d such that 2*d <= 511
    
    def score_mask(mask):
        """
        Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set.
        """
        count = 0
        for d in range(1, MAX_D + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def mask_to_list(mask):
        """Convert bitmask to list of integers."""
        res = []
        for i in range(1, MAX_VAL + 1):
            if (mask >> i) & 1:
                res.append(i)
        return res

    def list_to_mask(lst):
        """Convert list of integers to bitmask."""
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    # Try multiple random restarts with different initial strategies
    best_score = -1
    best_list = None
    
    # Strategy 1: Pure random
    # Strategy 2: Clustered (groups of consecutive numbers)
    # Strategy 3: Spread out (evenly spaced)
    
    strategies = []
    
    # Pure random
    strategies.append(random.sample(range(1, MAX_VAL + 1), N))
    
    # Clustered: 4 clusters of 8 consecutive numbers
    clusters = []
    positions = sorted(random.sample(range(1, MAX_VAL - 7), 4))
    for pos in positions:
        clusters.extend(range(pos, pos + 8))
    # Ensure exactly N elements and within range
    while len(clusters) < N:
        clusters.append(random.randint(1, MAX_VAL))
    clusters = list(set(clusters))[:N]
    if len(clusters) == N:
        strategies.append(clusters)
    
    # Spread out: roughly evenly spaced
    step = MAX_VAL // N
    spread = [1 + i * step + random.randint(0, max(1, step // 2)) for i in range(N)]
    spread = [min(x, MAX_VAL) for x in spread]
    spread = list(set(spread))
    while len(spread) < N:
        spread.append(random.randint(1, MAX_VAL))
    if len(set(spread)) == N:
        strategies.append(spread[:N])
    
    # Evaluate all strategies and pick the best as starting point
    best_init_score = -1
    best_init_list = strategies[0]
    for strat in strategies:
        if len(set(strat)) != N:
            continue
        m = list_to_mask(strat)
        s = score_mask(m)
        if s > best_init_score:
            best_init_score = s
            best_init_list = strat[:]
    
    current_list = best_init_list[:]
    current_mask = list_to_mask(current_list)
    current_score = best_init_score
    current_set = set(current_list)
    
    best_mask = current_mask
    best_score = current_score
    best_list = current_list[:]
    
    # Simulated Annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.9995
    
    no_improve_count = 0
    iteration = 0
    
    while time.time() < deadline:
        iteration += 1
        
        if iteration % 5000 == 0 and time.time() > deadline - 10.0:
            break
        
        # Generate neighbor: swap one element
        val_out = current_list[random.randint(0, N - 1)]
        
        # Pick element to add
        val_in = random.randint(1, MAX_VAL)
        attempts = 0
        while val_in in current_set and attempts < 100:
            val_in = random.randint(1, MAX_VAL)
            attempts += 1
        if val_in in current_set:
            continue
        
        # Apply move
        idx = current_list.index(val_out)
        current_list[idx] = val_in
        current_set.remove(val_out)
        current_set.add(val_in)
        
        new_mask = list_to_mask(current_list)
        new_score = score_mask(new_mask)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < pow(2.718281828, delta / T):
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
                best_list = current_list[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert
            current_list[idx] = val_out
            current_set.remove(val_in)
            current_set.add(val_out)
            no_improve_count += 1
        
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Restart with perturbation if stuck
        if no_improve_count > 3000:
            no_improve_count = 0
            perturb_count = random.randint(3, 6)
            for _ in range(perturb_count):
                rem_idx = random.randint(0, N - 1)
                val_to_remove = current_list[rem_idx]
                current_list.pop(rem_idx)
                current_set.discard(val_to_remove)
                
                new_val = random.randint(1, MAX_VAL)
                attempts = 0
                while new_val in current_set and attempts < 100:
                    new_val = random.randint(1, MAX_VAL)
                    attempts += 1
                if new_val not in current_set:
                    current_list.append(new_val)
                    current_set.add(new_val)
            
            current_mask = list_to_mask(current_list)
            current_score = score_mask(current_mask)
            T = max(T, 20.0)
    
    return best_list
# EVOLVE-BLOCK-END
