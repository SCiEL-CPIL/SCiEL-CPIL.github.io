import random
import time

def count_3aps(S):
    """Count distinct 3-term APs in set S."""
    S_set = set(S)
    count = 0
    for a in S:
        for b in S:
            if b > a:
                c = 2*b - a
                if c > b and c in S_set:
                    count += 1
    return count

def has_4ap(S):
    """Check if set S contains a nonconstant 4-term AP."""
    S_set = set(S)
    for a in S:
        for b in S:
            if b > a:
                d = b - a
                c = b + d
                e = c + d
                if e in S_set:
                    return True
    return False

def construct():
    n = 256
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit
    
    # Initial greedy construction: add x if it doesn't form a 4-AP
    # This is the seed approach but we'll try to improve it
    selected = set()
    for x in range(1, n + 1):
        # Check if adding x creates a 4-AP
        creates_4ap = False
        # x could be the 1st, 2nd, 3rd, or 4th term of a 4-AP
        # Case 1: x is 1st term: x, x+d, x+2d, x+3d
        for d in range(1, (n - x) // 3 + 1):
            if (x+d in selected) and (x+2*d in selected) and (x+3*d in selected):
                creates_4ap = True
                break
        if creates_4ap:
            continue
        # Case 2: x is 2nd term: x-d, x, x+d, x+2d
        for d in range(1, x):
            if (x-d in selected) and (x+d in selected) and (x+2*d in selected):
                creates_4ap = True
                break
        if creates_4ap:
            continue
        # Case 3: x is 3rd term: x-2d, x-d, x, x+d
        for d in range(1, x // 2 + 1):
            if (x-2*d in selected) and (x-d in selected) and (x+d in selected):
                creates_4ap = True
                break
        if creates_4ap:
            continue
        # Case 4: x is 4th term: x-3d, x-2d, x-d, x
        for d in range(1, x // 3 + 1):
            if (x-3*d in selected) and (x-2*d in selected) and (x-d in selected):
                creates_4ap = True
                break
        if creates_4ap:
            continue
        selected.add(x)
    
    current_3ap_count = count_3aps(selected)
    best_set = set(selected)
    best_count = current_3ap_count
    
    # Local search: try to swap elements to improve 3-AP count
    # while maintaining no 4-AP property
    random.seed(42)
    
    while time.time() - start_time < deadline:
        # Try random swaps
        improved = False
        for _ in range(500):
            if time.time() - start_time > deadline:
                break
            
            # Pick two elements: one to remove, one to add
            candidates_to_remove = list(selected)
            if not candidates_to_remove:
                break
            
            remove_elem = random.choice(candidates_to_remove)
            
            # Find candidates to add
            candidates_to_add = [x for x in range(1, n+1) if x not in selected]
            if not candidates_to_add:
                break
            
            add_elem = random.choice(candidates_to_add)
            
            # Try the swap
            trial_set = set(selected)
            trial_set.remove(remove_elem)
            trial_set.add(add_elem)
            
            # Check if trial set has no 4-AP
            if has_4ap(trial_set):
                continue
            
            # Count 3-APs
            trial_count = count_3aps(trial_set)
            
            if trial_count > current_3ap_count:
                selected = trial_set
                current_3ap_count = trial_count
                if current_3ap_count > best_count:
                    best_count = current_3ap_count
                    best_set = set(selected)
                improved = True
        
        if not improved:
            # Try single additions
            for _ in range(200):
                if time.time() - start_time > deadline:
                    break
                
                candidates_to_add = [x for x in range(1, n+1) if x not in selected]
                if not candidates_to_add:
                    break
                
                add_elem = random.choice(candidates_to_add)
                
                trial_set = set(selected)
                trial_set.add(add_elem)
                
                if has_4ap(trial_set):
                    continue
                
                trial_count = count_3aps(trial_set)
                
                if trial_count > current_3ap_count:
                    selected = trial_set
                    current_3ap_count = trial_count
                    if current_3ap_count > best_count:
                        best_count = current_3ap_count
                        best_set = set(selected)
                    break
            else:
                # Try single removals to enable better swaps
                for _ in range(100):
                    if time.time() - start_time > deadline:
                        break
                    
                    candidates_to_remove = list(selected)
                    if not candidates_to_remove:
                        break
                    
                    remove_elem = random.choice(candidates_to_remove)
                    
                    trial_set = set(selected)
                    trial_set.remove(remove_elem)
                    
                    trial_count = count_3aps(trial_set)
                    
                    # Only remove if it allows for a better configuration later
                    # For now, just track it
                    # We'll do a more sophisticated approach: remove and then try to add back
                    # multiple elements

    return sorted(best_set)
# EVOLVE-BLOCK-END
