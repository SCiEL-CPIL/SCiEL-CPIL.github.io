import time
import random

def construct():
    start_time = time.time()
    # Leave 5s for cleanup and return
    deadline = start_time + 85.0
    
    N = 1 << 12
    FULL_12 = (1 << 12) - 1
    
    # 1. Precompute comparability bitmasks
    # bits[i] is an integer with bit j set if i and j are comparable
    bits = [0] * N
    
    # subsets_mask[i] : bitmask of all subsets of i
    subsets_mask = [0] * N
    for i in range(N):
        m = 0
        sub = i
        while True:
            m |= (1 << sub)
            if sub == 0:
                break
            sub = (sub - 1) & i
        subsets_mask[i] = m
        
    # supersets_mask[i] : bitmask of all supersets of i
    supersets_mask = [0] * N
    for i in range(N):
        m = 0
        free = FULL_12 ^ i
        sub = free
        while True:
            j = i | sub
            m |= (1 << j)
            if sub == 0:
                break
            sub = (sub - 1) & free
        supersets_mask[i] = m
        
    for i in range(N):
        # Strict subset: subsets_mask[i] without i
        # Strict superset: supersets_mask[i] without i
        comp_mask = (subsets_mask[i] ^ (1 << i)) | (supersets_mask[i] ^ (1 << i))
        bits[i] = comp_mask

    def get_score(family_mask, family_list):
        s = 0
        for i in family_list:
            s += (bits[i] & family_mask).bit_count()
        return s // 2

    def greedy_construct():
        family = []
        family_mask = 0
        # We want to build a dense core.
        # Start with empty set?
        # Let's just greedy pick best candidate.
        for _ in range(40):
            best_x = -1
            best_score = -1
            for x in range(N):
                if (family_mask >> x) & 1:
                    continue
                score = (bits[x] & family_mask).bit_count()
                if score > best_score:
                    best_score = score
                    best_x = x
            family.append(best_x)
            family_mask |= (1 << best_x)
        return family, family_mask

    def simulated_annealing(init_family, init_mask, time_limit):
        current_family = init_family[:]
        current_mask = init_mask
        current_score = get_score(current_mask, current_family)
        
        T = 5.0
        cooling = 0.9995
        min_T = 0.01
        
        # To speed up, we can precompute a list of all indices
        # But random.randint is fast enough
        
        while time.time() < time_limit:
            idx = random.randint(0, len(current_family) - 1)
            u = current_family[idx]
            
            x = random.randint(0, N - 1)
            if x == u:
                continue
            if (current_mask >> x) & 1:
                continue
            
            mask_without_u = current_mask ^ (1 << u)
            
            gain = (bits[x] & mask_without_u).bit_count()
            loss = (bits[u] & mask_without_u).bit_count()
            delta = gain - loss
            
            if delta > 0:
                # Accept
                current_family[idx] = x
                current_mask = mask_without_u | (1 << x)
                current_score += delta
            else:
                # Accept with probability exp(delta/T)
                # delta is negative
                # Use random.random() < exp(delta/T)
                # To avoid overflow/underflow, compare log?
                # Or just use the formula. T is small, delta is small.
                if random.random() < pow(2.71828, delta / T):
                    current_family[idx] = x
                    current_mask = mask_without_u | (1 << x)
                    current_score += delta
                    
            T *= cooling
            if T < min_T:
                T = min_T
                
        return current_family, current_score

    best_family = None
    best_score = -1
    
    # Strategy 1: Greedy + SA
    fam1, mask1 = greedy_construct()
    score1 = get_score(mask1, fam1)
    if score1 > best_score:
        best_score = score1
        best_family = fam1
        
    time_left = deadline - time.time()
    if time_left > 5:
        fam_sa, score_sa = simulated_annealing(fam1, mask1, deadline - 2)
        if score_sa > best_score:
            best_score = score_sa
            best_family = fam_sa
            
    # Strategy 2: Random Inits + SA
    # Run a few random starts if time permits
    for i in range(5):
        if time.time() > deadline - 5:
            break
        random_family = random.sample(range(N), 40)
        random_mask = 0
        for x in random_family:
            random_mask |= (1 << x)
        
        fam_sa, score_sa = simulated_annealing(random_family, random_mask, deadline - 2)
        if score_sa > best_score:
            best_score = score_sa
            best_family = fam_sa
            
    # Convert to lists
    result = []
    for x in best_family:
        l = []
        for k in range(12):
            if (x >> k) & 1:
                l.append(k)
        result.append(l)
        
    return result
