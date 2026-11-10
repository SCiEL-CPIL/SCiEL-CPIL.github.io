import math
import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 135.0  # Safety margin before 150s kill
    
    def is_compatible(m1, r1, m2, r2):
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)
    
    def is_compatible_with_list(m, r, sel):
        for m2, r2 in sel:
            if not is_compatible(m, r, m2, r2):
                return False
        return True
    
    def calc_score(sel):
        return sum(1.0 / m for m, r in sel)
    
    # Greedy: process moduli in increasing order (highest 1/m first)
    # For each modulus, find a compatible residue
    selected = []
    
    for m in range(20, 201):
        found = False
        for r in range(m):
            if is_compatible_with_list(m, r, selected):
                selected.append((m, r))
                found = True
                break
        # If no residue works, skip this modulus
    
    best_score = calc_score(selected)
    best_selected = list(selected)
    
    # Local search to improve
    current = list(best_selected)
    current_score = best_score
    used_moduli = set(m for m, r in current)
    
    # Identify skipped moduli
    all_moduli = set(range(20, 201))
    skipped = list(all_moduli - used_moduli)
    
    attempts = 0
    max_attempts = 200000
    
    while time.time() - start_time < time_limit and attempts < max_attempts:
        attempts += 1
        
        # Try to add a skipped modulus with a compatible residue
        if skipped and random.random() < 0.5:
            m = random.choice(skipped)
            for r in range(m):
                if is_compatible_with_list(m, r, current):
                    current.append((m, r))
                    used_moduli.add(m)
                    skipped.remove(m)
                    current_score += 1.0 / m
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = list(current)
                    break
        
        # Try 2-opt: remove one selected, try to add two skipped
        elif len(current) > 2 and skipped and random.random() < 0.3:
            idx = random.randint(0, len(current) - 1)
            m_remove, r_remove = current.pop(idx)
            used_moduli.discard(m_remove)
            current_score -= 1.0 / m_remove
            
            # Try to add two from skipped
            added = []
            for m_add in random.sample(skipped, min(2, len(skipped))):
                for r in range(m_add):
                    if is_compatible_with_list(m_add, r, current):
                        current.append((m_add, r))
                        used_moduli.add(m_add)
                        skipped.remove(m_add)
                        current_score += 1.0 / m_add
                        added.append((m_add, r))
                        break
            
            if len(added) >= 2:
                if current_score > best_score:
                    best_score = current_score
                    best_selected = list(current)
            else:
                # Revert
                for m_add, r_add in added:
                    current.remove((m_add, r_add))
                    used_moduli.discard(m_add)
                    if m_add in skipped:
                        pass
                    else:
                        skipped.append(m_add)
                    current_score -= 1.0 / m_add
                current.append((m_remove, r_remove))
                used_moduli.add(m_remove)
                current_score += 1.0 / m_remove
                if m_remove not in skipped:
                    skipped.append(m_remove)
        
        # Try to change a residue to free up space for another modulus
        else:
            if len(current) > 1:
                idx = random.randint(0, len(current) - 1)
                m, r_old = current[idx]
                r_new = random.randint(0, m - 1)
                if r_new != r_old:
                    # Check compatibility with rest
                    compatible = True
                    for i, (m2, r2) in enumerate(current):
                        if i == idx:
                            continue
                        if not is_compatible(m, r_new, m2, r2):
                            compatible = False
                            break
                    if compatible:
                        current[idx] = (m, r_new)
    
    # Final polish: try to add any remaining skipped moduli deterministically
    for m in sorted(skipped, key=lambda x: 1.0/x, reverse=True):
        for r in range(m):
            if is_compatible_with_list(m, r, current):
                current.append((m, r))
                current_score += 1.0 / m
                if current_score > best_score:
                    best_score = current_score
                    best_selected = list(current)
                break
    
    return [[m, r] for m, r in best_selected]
