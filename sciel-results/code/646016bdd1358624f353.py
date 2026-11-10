import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 12
    m = 40
    num_masks = 1 << n
    full_mask = num_masks - 1
    
    # Precompute comparability bitsets
    # comp_bits[i] is a big integer where bit j is 1 if i and j are comparable
    # (i subset j or j subset i). Note: i is always comparable to itself.
    
    comp_bits = [0] * num_masks
    
    # Generate submasks for each i
    # If j is a submask of i, then j <= i.
    # We iterate submasks of i.
    for i in range(num_masks):
        sub = i
        while True:
            # sub is a submask of i
            # Set bit sub in comp_bits[i]
            comp_bits[i] |= (1 << sub)
            # Set bit i in comp_bits[sub]
            comp_bits[sub] |= (1 << i)
            
            if sub == 0:
                break
            sub = (sub - 1) & i
            
    # Helper to compute degrees
    def get_degrees(masks, S_mask_val):
        degs = []
        for mask in masks:
            deg = (comp_bits[mask] & S_mask_val).bit_count() - 1
            degs.append(deg)
        return degs

    # Helper to compute score
    def compute_score(masks, S_mask_val):
        degs = get_degrees(masks, S_mask_val)
        return sum(degs) // 2

    # Initialization Strategy:
    # We want a dense structure. 
    # A chain of length 12 has 11 comparable pairs per chain, but only 12 sets.
    # We need 40 sets.
    # Let's try to construct multiple chains or a lattice-like structure.
    
    # Random restarts with different initializations might be good.
    # But let's start with a greedy construction that maximizes immediate gains.
    
    # Greedy Init:
    # Start with empty set and full set? No, start with random or specific.
    # Let's start with the empty set (0) and full set (4095).
    # Then greedily add sets that maximize comparability with existing sets.
    
    def greedy_init():
        # Start with a few anchor points
        # 0 and 4095 are comparable to everything.
        current_masks = [0, 4095]
        used_masks = set(current_masks)
        S_mask = (1 << 0) | (1 << 4095)
        
        for _ in range(m - len(current_masks)):
            best_mask = -1
            best_score = -1
            # To speed up, we can sample or iterate all.
            # Iterating all 4096 is fast enough for 38 steps.
            for v in range(num_masks):
                if v in used_masks:
                    continue
                score = (comp_bits[v] & S_mask).bit_count()
                if score > best_score:
                    best_score = score
                    best_mask = v
            if best_mask != -1:
                current_masks.append(best_mask)
                used_masks.add(best_mask)
                S_mask |= (1 << best_mask)
                
        return current_masks

    current_masks = greedy_init()
    used_masks = set(current_masks)
    S_mask = 0
    for mask in current_masks:
        S_mask |= (1 << mask)
        
    degs = get_degrees(current_masks, S_mask)
    current_score = sum(degs) // 2
    best_score = current_score
    best_solution = list(current_masks)
    
    # Local Search with Simulated Annealing
    # State: current_masks, degs, S_mask, used_masks
    
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.1
    
    no_improvement_count = 0
    steps_since_restart = 0
    
    while time.time() < deadline:
        # Check time
        if time.time() > deadline - 5.0:
            break
            
        # Pick a random index u
        u_idx = random.randint(0, m - 1)
        u_mask = current_masks[u_idx]
        u_deg = degs[u_idx]
        
        # Pick a random candidate v not in used_masks
        # To avoid expensive list construction, we can try random v's
        # or maintain a list of unused. 
        # Number of unused is 4096 - 40 = 4056.
        # Random sampling might hit used ones often? No, 40/4096 is small.
        # Let's just pick random v in range(num_masks) and check.
        
        v = -1
        for _ in range(10): # Try up to 10 random picks
            candidate = random.randint(0, num_masks - 1)
            if candidate not in used_masks:
                v = candidate
                break
        
        if v == -1:
            # Fallback: find one
            for candidate in range(num_masks):
                if candidate not in used_masks:
                    v = candidate
                    break
            if v == -1:
                break # Should not happen
                
        # Calculate delta
        S_mask_without_u = S_mask ^ (1 << u_mask)
        v_deg_in_S_minus_u = (comp_bits[v] & S_mask_without_u).bit_count()
        delta = v_deg_in_S_minus_u - u_deg
        
        # Acceptance criterion
        accept = False
        if delta >= 0:
            accept = True
        else:
            # SA: accept with prob exp(delta / T)
            # delta is negative
            r = random.random()
            if r < math.exp(delta / T):
                accept = True
                
        if accept:
            # Perform swap
            old_u = u_mask
            new_v = v
            
            # Update S_mask
            S_mask ^= (1 << old_u)
            S_mask |= (1 << new_v)
            
            # Update used_masks
            used_masks.remove(old_u)
            used_masks.add(new_v)
            
            # Update current_masks
            current_masks[u_idx] = new_v
            
            # Update degs
            # For all j != u_idx:
            # If old_u was comparable to S[j], degs[j] -= 1
            # If new_v is comparable to S[j], degs[j] += 1
            
            for j in range(m):
                if j == u_idx:
                    continue
                j_mask = current_masks[j]
                if (comp_bits[old_u] >> j_mask) & 1:
                    degs[j] -= 1
                if (comp_bits[new_v] >> j_mask) & 1:
                    degs[j] += 1
            
            # Update degs[u_idx]
            degs[u_idx] = (comp_bits[new_v] & S_mask).bit_count() - 1
            
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_solution = list(current_masks)
                no_improvement_count = 0
                steps_since_restart = 0
            else:
                no_improvement_count += 1
                steps_since_restart += 1
                
            # Cooling
            T = max(min_T, T * cooling_rate)
            
            # Restart logic
            if no_improvement_count > 1000:
                # Reheat
                T = 10.0
                no_improvement_count = 0
                
                # Optional: Random perturbation if stuck for too long
                if steps_since_restart > 5000:
                    # Perform 5 random swaps without checking delta
                    for _ in range(5):
                        ru_idx = random.randint(0, m - 1)
                        ru_mask = current_masks[ru_idx]
                        
                        rv = -1
                        for _ in range(10):
                            candidate = random.randint(0, num_masks - 1)
                            if candidate not in used_masks:
                                rv = candidate
                                break
                        if rv == -1:
                            continue
                        
                        # Swap
                        old_ru = ru_mask
                        new_rv = rv
                        
                        S_mask ^= (1 << old_ru)
                        S_mask |= (1 << new_rv)
                        used_masks.remove(old_ru)
                        used_masks.add(new_rv)
                        current_masks[ru_idx] = new_rv
                        
                        for j in range(m):
                            if j == ru_idx:
                                continue
                            j_mask = current_masks[j]
                            if (comp_bits[old_ru] >> j_mask) & 1:
                                degs[j] -= 1
                            if (comp_bits[new_rv] >> j_mask) & 1:
                                degs[j] += 1
                            
                        degs[ru_idx] = (comp_bits[new_rv] & S_mask).bit_count() - 1
                        
                    # Recalculate score
                    current_score = compute_score(current_masks, S_mask)
                    steps_since_restart = 0
                    no_improvement_count = 0

    # Convert best_solution to list of lists
    result = []
    for mask in best_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result

