"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_4ap_free(s):
    """Check if set s contains no 4-term arithmetic progression."""
    s_set = set(s)
    max_val = max(s_set) if s_set else 0
    # Iterate through all elements as potential 4th term (largest)
    # a, a+d, a+2d, a+3d
    # x = a+3d => a = x-3d, a+d = x-2d, a+2d = x-d
    for x in s_set:
        # d must be such that x-3d >= 1
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - d) in s_set and (x - 2*d) in s_set and (x - 3*d) in s_set:
                return False
    return True

def _get_conflicts(s, x):
    """
    Return a list of 4-APs in s U {x} that contain x.
    Each AP is a tuple of 4 elements.
    """
    s_set = set(s)
    conflicts = []
    
    # Case 1: x is the 4th term (largest)
    # x-3d, x-2d, x-d, x
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if (x - d) in s_set and (x - 2*d) in s_set and (x - 3*d) in s_set:
            conflicts.append((x - 3*d, x - 2*d, x - d, x))
            
    # Case 2: x is the 3rd term
    # x-2d, x-d, x, x+d
    # Need x+d in s
    if x + 1 <= 512:
        # d can be 1..?
        # x-2d >= 1 => d <= (x-1)//2
        max_d = (x - 1) // 2
        for d in range(1, max_d + 1):
            if (x + d) in s_set and (x - d) in s_set and (x - 2*d) in s_set:
                conflicts.append((x - 2*d, x - d, x, x + d))
                
    # Case 3: x is the 2nd term
    # x-d, x, x+d, x+2d
    # Need x+d, x+2d in s
    # x-d >= 1 => d <= x-1
    # x+2d <= 512
    max_d = min(x - 1, (512 - x) // 2)
    for d in range(1, max_d + 1):
        if (x - d) in s_set and (x + d) in s_set and (x + 2*d) in s_set:
            conflicts.append((x - d, x, x + d, x + 2*d))
            
    # Case 4: x is the 1st term (smallest)
    # x, x+d, x+2d, x+3d
    # Need x+d, x+2d, x+3d in s
    # x+3d <= 512 => d <= (512-x)//3
    max_d = (512 - x) // 3
    for d in range(1, max_d + 1):
        if (x + d) in s_set and (x + 2*d) in s_set and (x + 3*d) in s_set:
            conflicts.append((x, x + d, x + 2*d, x + 3*d))
            
    return conflicts

def construct():
    deadline = time.time() + 55.0  # Safety margin
    
    # 1. Greedy Initialization
    selected_set = set()
    for x in range(1, 513):
        is_safe = True
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - d) in selected_set and (x - 2*d) in selected_set and (x - 3*d) in selected_set:
                is_safe = False
                break
        if is_safe:
            selected_set.add(x)
            
    # 2. Local Search (Hill Climbing)
    # We try to add elements. If blocked, we try to remove an element from the conflict to allow the add.
    # We also try "Remove 1, Add 2" moves if simple adds fail.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add any element
        candidates = [x for x in range(1, 513) if x not in selected_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
                
            # Check if adding x is valid
            if _is_4ap_free(selected_set | {x}):
                selected_set.add(x)
                improved = True
                break # Restart loop with new set
                
            # If not valid, find conflicts
            conflicts = _get_conflicts(selected_set, x)
            if not conflicts:
                continue
                
            # Try to resolve by removing one element from one of the conflicts
            # We pick a random conflict
            ap = random.choice(conflicts)
            # The element x is in the AP. We want to remove one of the OTHER elements.
            others = [y for y in ap if y != x]
            random.shuffle(others)
            
            for y in others:
                # Tentatively remove y and add x
                new_set = (selected_set - {y}) | {x}
                if _is_4ap_free(new_set):
                    selected_set = new_set
                    improved = True
                    break # Restart loop
                    
            if improved:
                break
                
        # If no simple add/swap improved the set, try a "Remove 1, Add 2" move
        if not improved and time.time() < deadline:
            # Pick a random element to remove
            if selected_set:
                y = random.choice(list(selected_set))
                temp_set = selected_set - {y}
                
                # Try to add as many elements as possible to temp_set
                # We only care if we can add at least 2 to increase total size by 1 (remove 1, add 2 => net +1)
                # Or even just add 1? No, remove 1 add 1 is net 0. We want net +1.
                # So we need to add at least 2 elements to the set after removing y.
                
                added = 0
                new_set = set(temp_set)
                candidates = [x for x in range(1, 513) if x not in new_set]
                random.shuffle(candidates)
                
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if _is_4ap_free(new_set | {x}):
                        new_set.add(x)
                        added += 1
                        if added >= 2:
                            # We have a better set
                            selected_set = new_set
                            improved = True
                            break

    return sorted(selected_set)
# EVOLVE-BLOCK-END
