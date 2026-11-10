import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 590  # safety margin

    # 1. Compute Forbidden Differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build Graph as Bitmasks
    # adj_mask[i] has bit j set if i and j are connected (forbidden difference)
    adj_mask = [0] * m
    for i in range(m):
        for j in range(i + 1, m):
            d = (i - j) % m
            if d in forbidden:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                
    full_mask = (1 << m) - 1
    
    # Precompute degrees for greedy heuristic
    degrees = [bin(adj_mask[i]).count("1") for i in range(m)]
    
    def get_size(mask):
        return bin(mask).count("1")
        
    def greedy_init(shuffle_seed, strategy="min_degree"):
        rng = random.Random(shuffle_seed)
        candidates = list(range(m))
        
        if strategy == "random":
            rng.shuffle(candidates)
        else:
            # Sort by degree (ascending), then randomize ties?
            # To keep it diverse, we can add random noise to degree
            scored = [(degrees[v] + rng.random() * 2, v) for v in candidates]
            scored.sort()
            candidates = [v for _, v in scored]
            
        selected_mask = 0
        for v in candidates:
            if (adj_mask[v] & selected_mask) == 0:
                selected_mask |= (1 << v)
        return selected_mask

    def local_improve(mask, rng):
        changed = True
        while changed:
            changed = False
            addable = []
            not_selected = full_mask ^ mask
            # Iterate through all v
            for v in range(m):
                if (not_selected >> v) & 1:
                    if (adj_mask[v] & mask) == 0:
                        addable.append(v)
                        
            if not addable:
                break
                
            # Pick one to add
            # Heuristic: pick the one with minimum degree among addable?
            # Or random?
            # Let's try minimum degree to leave room for others
            if len(addable) > 1:
                # Score by degree
                v = min(addable, key=lambda x: degrees[x])
                # Add some randomness
                if rng.random() < 0.2:
                    v = rng.choice(addable)
            else:
                v = addable[0]
                
            mask |= (1 << v)
            changed = True
            
        return mask

    def perturb(mask, rng, k_remove):
        vertices = [v for v in range(m) if (mask >> v) & 1]
        if len(vertices) < k_remove:
            return 0
        to_remove = rng.sample(vertices, k_remove)
        new_mask = mask
        for v in to_remove:
            new_mask &= ~(1 << v)
        return new_mask

    best_mask = 0
    best_size = 0
    
    # Initial Greedy Runs
    # Try a mix of strategies
    for i in range(20):
        m_init = greedy_init(i, "min_degree")
        m_init = local_improve(m_init, random.Random(i))
        if get_size(m_init) > best_size:
            best_size = get_size(m_init)
            best_mask = m_init
        if time.time() > deadline:
            return np.array([v for v in range(m) if (best_mask >> v) & 1], dtype=np.int64)

    for i in range(20):
        m_init = greedy_init(i + 100, "random")
        m_init = local_improve(m_init, random.Random(i + 100))
        if get_size(m_init) > best_size:
            best_size = get_size(m_init)
            best_mask = m_init
        if time.time() > deadline:
            return np.array([v for v in range(m) if (best_mask >> v) & 1], dtype=np.int64)

    # ILS Loop
    rng = random.Random(42)
    curr_mask = best_mask
    curr_size = best_size
    temp = 50.0
    min_temp = 1.0
    cooling = 0.99
    
    # Track stagnation to restart
    stagnation_counter = 0
    stagnation_limit = 50
    
    while time.time() < deadline:
        # Perturb
        # Adaptive k_remove: larger if stuck
        k_remove = rng.randint(3, 10)
        if stagnation_counter > stagnation_limit // 2:
            k_remove += rng.randint(0, 5)
            
        pert_mask = perturb(curr_mask, rng, k_remove)
        
        # Local Improve
        new_mask = local_improve(pert_mask, rng)
        new_size = get_size(new_mask)
        
        # Acceptance
        if new_size >= curr_size:
            curr_mask = new_mask
            curr_size = new_size
            stagnation_counter = 0
            if new_size > best_size:
                best_size = new_size
                best_mask = new_mask
        else:
            delta = new_size - curr_size
            # Metropolis
            if rng.random() < pow(2.71828, delta / temp):
                curr_mask = new_mask
                curr_size = new_size
            stagnation_counter += 1
            
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
            
        # If stuck for too long, restart from best
        if stagnation_counter > stagnation_limit:
            curr_mask = best_mask
            curr_size = best_size
            temp = 50.0
            stagnation_counter = 0

    # Final Polish: Try all single additions and simple swaps on best_mask
    # Just in case the ILS converged to a non-maximal set
    final_mask = local_improve(best_mask, rng)
    
    # Convert to array
    result = []
    for v in range(m):
        if (final_mask >> v) & 1:
            result.append(v)
            
    return np.array(result, dtype=np.int64)
