import math
import random
import time

def construct():
    start_time = time.time()
    time_limit = 140.0
    
    # Precompute gcds
    moduli = list(range(20, 201))
    gcds = {}
    for i, m1 in enumerate(moduli):
        for j, m2 in enumerate(moduli):
            if i <= j: # Symmetric
                gcds[(m1, m2)] = math.gcd(m1, m2)
                gcds[(m2, m1)] = gcds[(m1, m2)]
                
    def is_compatible(m1, r1, m2, r2):
        g = gcds[(m1, m2)]
        return (r1 % g) != (r2 % g)
        
    def is_compatible_with_list(m, r, selected):
        for m2, r2 in selected:
            if not is_compatible(m, r, m2, r2):
                return False
        return True
        
    def calc_score(selected):
        return sum(1.0 / m for m, r in selected)

    # 1. Greedy Initialization
    selected = []
    for m in moduli:
        for r in range(m):
            if is_compatible_with_list(m, r, selected):
                selected.append((m, r))
                break
                
    best_selected = list(selected)
    best_score = calc_score(best_selected)
    
    current = list(selected)
    current_score = best_score
    all_moduli = set(moduli)
    
    def try_add_modulus(m, cur_set):
        for r in range(m):
            if is_compatible_with_list(m, r, cur_set):
                return r
        return None

    # 2. Targeted Local Search
    stuck_count = 0
    max_stuck = 20000
    
    while time.time() - start_time < time_limit and stuck_count < max_stuck:
        
        if time.time() - start_time > time_limit:
            break

        # 1. Try to add any missing modulus
        missing = list(all_moduli - set(m for m, r in current))
        missing.sort() # Smallest first
        
        added = False
        for m in missing:
            r = try_add_modulus(m, current)
            if r is not None:
                current.append((m, r))
                current_score += 1.0 / m
                added = True
                if current_score > best_score:
                    best_score = current_score
                    best_selected = list(current)
                break
        
        if added:
            stuck_count = 0
            continue
            
        # 2. If stuck, try to unblock the smallest missing modulus
        if not missing:
            break
            
        target_m = missing[0]
        
        # Find which residues of target_m are blocked by which existing moduli
        blocked_by = {r: [] for r in range(target_m)}
        
        for idx, (m2, r2) in enumerate(current):
            for r in range(target_m):
                if not is_compatible(target_m, r, m2, r2):
                    blocked_by[r].append(idx)
                    
        min_blockers = min(len(blocked_by[r]) for r in range(target_m))
        
        if min_blockers == 0:
            r = try_add_modulus(target_m, current)
            if r is not None:
                current.append((target_m, r))
                current_score += 1.0 / target_m
                if current_score > best_score:
                    best_score = current_score
                    best_selected = list(current)
                stuck_count = 0
                continue
        
        candidates = [r for r in range(target_m) if len(blocked_by[r]) == min_blockers]
        r_star = random.choice(candidates)
        blockers = blocked_by[r_star]
        
        blocker_idx = random.choice(blockers)
        m_blocker, r_blocker = current[blocker_idx]
        
        found_new_res = False
        for r_new in range(m_blocker):
            if r_new == r_blocker:
                continue
            
            if not is_compatible(m_blocker, r_new, target_m, r_star):
                continue
                
            compatible = True
            for i, (m2, r2) in enumerate(current):
                if i == blocker_idx:
                    continue
                if not is_compatible(m_blocker, r_new, m2, r2):
                    compatible = False
                    break
                    
            if compatible:
                current[blocker_idx] = (m_blocker, r_new)
                found_new_res = True
                break
                
        if found_new_res:
            stuck_count = 0 
            continue
            
        if random.random() < 0.5 and len(current) > 1:
             idx = random.randint(0, len(current) - 1)
             m, r_old = current[idx]
             r_new = random.randint(0, m - 1)
             if r_new != r_old:
                 compatible = True
                 for i, (m2, r2) in enumerate(current):
                     if i == idx: continue
                     if not is_compatible(m, r_new, m2, r2):
                         compatible = False
                         break
                 if compatible:
                     current[idx] = (m, r_new)
                     
        stuck_count += 1

    # Final Polish
    for m in sorted(all_moduli - set(m for m, r in current), key=lambda x: 1.0/x, reverse=True):
        r = try_add_modulus(m, current)
        if r is not None:
            current.append((m, r))
            current_score += 1.0 / m
            if current_score > best_score:
                best_score = current_score
                best_selected = list(current)
                
    return [[m, r] for m, r in best_selected]
