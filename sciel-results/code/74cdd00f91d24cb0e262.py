import time
import random
import math

def construct():
    deadline = time.time() + 55
    N = 256
    
    # Precomputation
    # Elements are 1..N, indices 0..N-1
    three_aps = []
    four_aps = []
    
    # Generate 3-APs
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a-1, a+d-1, a+2*d-1))
            
    # Generate 4-APs
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a-1, a+d-1, a+2*d-1, a+3*d-1))
            
    # Mappings
    elem_to_3aps = [[] for _ in range(N)]
    for i, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3aps[x].append(i)
            
    elem_to_4aps = [[] for _ in range(N)]
    for i, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(i)
            
    # Bitmasks for each AP
    mask_3ap = [0] * len(three_aps)
    for i, ap in enumerate(three_aps):
        m = 0
        for x in ap:
            m |= (1 << x)
        mask_3ap[i] = m
        
    mask_4ap = [0] * len(four_aps)
    for i, ap in enumerate(four_aps):
        m = 0
        for x in ap:
            m |= (1 << x)
        mask_4ap[i] = m

    # Helper to calculate delta score/violations for adding x to S
    # S is current bitmask
    # Returns (delta_score, delta_violations)
    def calc_add(S, x):
        bit_x = 1 << x
        d_score = 0
        d_viol = 0
        
        # Check 3-APs completed by x
        for ap_idx in elem_to_3aps[x]:
            m = mask_3ap[ap_idx]
            # Check if other 2 elements are in S
            others = m & ~bit_x
            if (S & others) == others:
                d_score += 1
                
        # Check 4-APs completed by x
        for ap_idx in elem_to_4aps[x]:
            m = mask_4ap[ap_idx]
            others = m & ~bit_x
            if (S & others) == others:
                d_viol += 1
                
        return d_score, d_viol

    # Helper to calculate delta score/violations for removing x from S
    def calc_remove(S, x):
        bit_x = 1 << x
        d_score = 0
        d_viol = 0
        
        for ap_idx in elem_to_3aps[x]:
            m = mask_3ap[ap_idx]
            others = m & ~bit_x
            if (S & others) == others:
                d_score -= 1
                
        for ap_idx in elem_to_4aps[x]:
            m = mask_4ap[ap_idx]
            others = m & ~bit_x
            if (S & others) == others:
                d_viol -= 1
                
        return d_score, d_viol

    # Initialization: Greedy Random
    random.seed(42)
    S = 0
    score = 0
    violations = 0
    
    elements = list(range(N))
    random.shuffle(elements)
    
    for x in elements:
        if S & (1 << x):
            continue
        d_s, d_v = calc_add(S, x)
        if d_v == 0:
            S |= (1 << x)
            score += d_s
            violations += d_v # remains 0
            
    best_S = S
    best_score = score
    best_violations = violations
    best_fitness = best_score - 100 * best_violations

    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    penalty = 100.0
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        # Choose move
        move_type = random.random()
        
        if move_type < 0.4: # Add
            # Find a random element not in S
            # Efficient way: pick random index, if in S, retry (max few times) or use a list
            # Since N=256, list of available elements is fast enough
            available = [i for i in range(N) if not (S & (1 << i))]
            if not available:
                continue
            x = random.choice(available)
            
            d_s, d_v = calc_add(S, x)
            new_S = S | (1 << x)
            new_score = score + d_s
            new_violations = violations + d_v
            new_fitness = new_score - penalty * new_violations
            old_fitness = score - penalty * violations
            
            delta_fit = new_fitness - old_fitness
            if delta_fit > 0 or (delta_fit < 0 and random.random() < math.exp(delta_fit / T)):
                S = new_S
                score = new_score
                violations = new_violations
                if violations == 0 and score > best_score:
                    best_S = S
                    best_score = score
                    best_violations = 0
                    best_fitness = best_score
                    
        elif move_type < 0.7: # Remove
            # Find a random element in S
            # Count bits or iterate
            # Iteration is fine for 256
            in_set = [i for i in range(N) if (S & (1 << i))]
            if not in_set:
                continue
            x = random.choice(in_set)
            
            d_s, d_v = calc_remove(S, x)
            new_S = S & ~(1 << x)
            new_score = score + d_s
            new_violations = violations + d_v
            new_fitness = new_score - penalty * new_violations
            old_fitness = score - penalty * violations
            
            delta_fit = new_fitness - old_fitness
            if delta_fit > 0 or (delta_fit < 0 and random.random() < math.exp(delta_fit / T)):
                S = new_S
                score = new_score
                violations = new_violations
                if violations == 0 and score > best_score:
                    best_S = S
                    best_score = score
                    best_violations = 0
                    best_fitness = best_score
                    
        else: # Swap
            in_set = [i for i in range(N) if (S & (1 << i))]
            not_in_set = [i for i in range(N) if not (S & (1 << i))]
            
            if not in_set or not not_in_set:
                continue
                
            x_out = random.choice(in_set)
            x_in = random.choice(not_in_set)
            
            # Remove x_out
            d_s_out, d_v_out = calc_remove(S, x_out)
            # Add x_in to (S without x_out)
            S_temp = S & ~(1 << x_out)
            d_s_in, d_v_in = calc_add(S_temp, x_in)
            
            new_S = S_temp | (1 << x_in)
            new_score = score + d_s_out + d_s_in
            new_violations = violations + d_v_out + d_v_in
            new_fitness = new_score - penalty * new_violations
            old_fitness = score - penalty * violations
            
            delta_fit = new_fitness - old_fitness
            if delta_fit > 0 or (delta_fit < 0 and random.random() < math.exp(delta_fit / T)):
                S = new_S
                score = new_score
                violations = new_violations
                if violations == 0 and score > best_score:
                    best_S = S
                    best_score = score
                    best_violations = 0
                    best_fitness = best_score
                    
        T *= cooling

    # Final Repair: Ensure no violations
    # If best_S has violations, we need to remove elements
    # But we tracked best_S only when violations==0. 
    # However, the current S might have violations. 
    # The requirement is to return a valid set. 
    # best_S is guaranteed valid by construction (only updated if violations==0).
    # But what if best_S is empty? 
    # Greedy init ensures non-empty usually.
    
    # Just in case, if best_S has violations (shouldn't happen), repair it
    if best_violations > 0:
        # Repair best_S
        S_rep = best_S
        score_rep = best_score
        viol_rep = best_violations
        
        while viol_rep > 0:
            # Find a violating 4-AP
            found = False
            for i, ap in enumerate(four_aps):
                m = mask_4ap[i]
                if (S_rep & m) == m:
                    # This AP is violated. Remove one element from it.
                    # Pick the one with lowest score contribution
                    min_loss = float('inf')
                    min_x = -1
                    for x in ap:
                        # Calculate score loss if we remove x
                        # Approximate: count 3-APs containing x that are currently satisfied
                        # For speed, just remove random one or first one
                        # Let's be a bit smarter: remove x that is in fewest 3-APs
                        # Or just random
                        # To be safe and fast, remove random
                        min_x = random.choice(ap)
                        break
                    # Remove min_x
                    S_rep &= ~(1 << min_x)
                    # Update score and violations
                    # Recalculate is safer for correctness in repair phase
                    # But let's just recalculate score for S_rep
                    score_rep = 0
                    viol_rep = 0
                    for i3, ap3 in enumerate(three_aps):
                        if (S_rep & mask_3ap[i3]) == mask_3ap[i3]:
                            score_rep += 1
                    for i4, ap4 in enumerate(four_aps):
                        if (S_rep & mask_4ap[i4]) == mask_4ap[i4]:
                            viol_rep += 1
                    found = True
                    break
            if not found:
                break
                
        best_S = S_rep
        best_score = score_rep
        best_violations = viol_rep

    # Convert bitmask to sorted list of 1-based indices
    result = []
    for i in range(N):
        if best_S & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
