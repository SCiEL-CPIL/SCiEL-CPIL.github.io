import math
import random
import time

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    # Generate candidates
    candidates = []
    for m in range(20, 201):
        for r in range(m):
            candidates.append((m, r))
            
    # Helper: Compatibility
    def is_compatible(m1, r1, m2, r2):
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)

    def is_compatible_with_all(m, r, selected_list):
        for m2, r2 in selected_list:
            if not is_compatible(m, r, m2, r2):
                return False
        return True

    # 1. Greedy Initialization
    selected = []
    selected_set = set()
    score = 0.0
    
    # Sort by m ascending (highest weight)
    for m in range(20, 201):
        # Try to find a compatible residue
        # Randomize residue order to break ties/symmetries? 
        # Or just 0..m-1.
        # Let's try 0..m-1.
        for r in range(m):
            if is_compatible_with_all(m, r, selected):
                selected.append((m, r))
                selected_set.add((m, r))
                score += 1.0 / m
                break

    best_selected = list(selected)
    best_score = score

    # 2. Local Search
    # We will try to improve the score.
    # Moves: Add, Remove, Swap Residue.
    
    # To speed up compatibility checks, we can't easily index by m if we have multiple m? 
    # No, moduli are distinct. So we can have a dict m -> r for selected.
    # But we need to check against ALL.
    
    # Let's use the list-based check. N is small.
    
    current_selected = list(best_selected)
    current_set = set(best_selected)
    current_score = best_score
    
    # Random seed for reproducibility in debugging, but we want exploration.
    # random.seed(42)
    
    while time.time() - start_time < time_limit:
        
        # Phase 1: Try to Add
        # Try random candidates
        added = False
        # Limit attempts to avoid wasting time if set is very dense
        for _ in range(200): 
            if time.time() - start_time > time_limit:
                break
                
            m, r = random.choice(candidates)
            if (m, r) in current_set:
                continue
                
            if is_compatible_with_all(m, r, current_selected):
                current_selected.append((m, r))
                current_set.add((m, r))
                current_score += 1.0 / m
                added = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_selected = list(current_selected)
        
        if not added:
            # Phase 2: Kick
            if not current_selected:
                break
                
            move_type = random.random()
            
            if move_type < 0.5:
                # Remove a random element
                idx = random.randint(0, len(current_selected) - 1)
                m, r = current_selected.pop(idx)
                current_set.remove((m, r))
                current_score -= 1.0 / m
            else:
                # Change residue of a random element
                idx = random.randint(0, len(current_selected) - 1)
                m, r_old = current_selected[idx]
                r_new = random.randint(0, m - 1)
                
                if r_new == r_old:
                    continue
                    
                # Check if new residue is compatible with others
                # Temporarily remove
                current_set.remove((m, r_old))
                
                if is_compatible_with_all(m, r_new, current_selected):
                    current_selected[idx] = (m, r_new)
                    current_set.add((m, r_new))
                else:
                    # Revert
                    current_selected[idx] = (m, r_old)
                    current_set.add((m, r_old))
                    
                # Score doesn't change on residue change
                # But we might enable additions in the next loop iteration

    return [[m, r] for m, r in best_selected]
