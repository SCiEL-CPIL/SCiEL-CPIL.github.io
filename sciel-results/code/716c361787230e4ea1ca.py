import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    n = 12
    m = 40
    total_masks = 1 << n
    
    def is_comp(a, b):
        return (a & b) == a or (a & b) == b

    def greedy_init():
        family = [0]
        used = {0}
        
        for _ in range(m - 1):
            best_score = -1
            best_mask = -1
            # Optimization: List of candidates? No, just iterate.
            for mask in range(total_masks):
                if mask in used:
                    continue
                cnt = 0
                for x in family:
                    if is_comp(mask, x):
                        cnt += 1
                if cnt > best_score:
                    best_score = cnt
                    best_mask = mask
            family.append(best_mask)
            used.add(best_mask)
            
        return family

    def local_search(family, time_limit):
        # family is list of ints
        comp_counts = [0] * m
        for i in range(m):
            for j in range(m):
                if i != j and is_comp(family[i], family[j]):
                    comp_counts[i] += 1
        
        current_score = sum(comp_counts) // 2
        best_family = family[:]
        best_score = current_score
        
        T = 5.0
        cooling = 0.9999
        min_T = 0.01
        
        # Use local vars for speed
        m_local = m
        total_local = total_masks
        is_comp_local = is_comp
        random_int = random.randint
        random_rand = random.random
        exp_local = math.exp
        
        while time.time() < time_limit:
            i = random_int(0, m_local - 1)
            old_mask = family[i]
            
            # Find new mask
            while True:
                new_mask = random_int(0, total_local - 1)
                if new_mask not in family:
                    break
            
            # Calc new_contrib
            new_contrib = 0
            for j in range(m_local):
                if j == i:
                    continue
                if is_comp_local(new_mask, family[j]):
                    new_contrib += 1
            
            delta = new_contrib - comp_counts[i]
            
            accept = False
            if delta > 0:
                accept = True
            else:
                # exp(delta / T)
                # delta is negative
                # If T is very small, exp is 0
                if T > min_T:
                    prob = exp_local(delta / T)
                else:
                    prob = 0.0
                if random_rand() < prob:
                    accept = True
            
            if accept:
                family[i] = new_mask
                current_score += delta
                
                # Update comp_counts
                for j in range(m_local):
                    if j == i:
                        continue
                    old_rel = is_comp_local(old_mask, family[j])
                    new_rel = is_comp_local(new_mask, family[j])
                    if new_rel != old_rel:
                        if new_rel:
                            comp_counts[j] += 1
                        else:
                            comp_counts[j] -= 1
                comp_counts[i] = new_contrib
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
                
                if T > min_T:
                    T *= cooling
            else:
                # No update to family or scores
                # Still cool down? Usually yes in SA.
                if T > min_T:
                    T *= cooling
            
        return best_family

    # Run
    family = greedy_init()
    
    # If time remains, we could do another init, but one is usually enough
    # given the long SA phase.
    
    final_family = local_search(family, deadline)
    
    # Convert to lists
    result = []
    for mask in final_family:
        result.append([b for b in range(n) if (mask >> b) & 1])
        
    return result
