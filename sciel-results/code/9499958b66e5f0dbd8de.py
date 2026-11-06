import random

def has_4ap_with(set_s, x):
    """Check if adding x to set_s creates a 4-term AP."""
    s = set_s | {x}
    # A 4-AP: a, a+d, a+2d, a+3d with d>0
    # x could be any of the four positions
    # Case 1: x = a+3d => a, a+d, a+2d in s, a = x-3d
    # Case 2: x = a+2d => a, a+d in s, a+3d in s; a = x-2d
    # Case 3: x = a+d => a in s, a+2d, a+3d in s; a = x-d
    # Case 4: x = a => a+d, a+2d, a+3d in s
    for d in range(1, 513):
        # x as 4th term: a = x-3d
        a = x - 3*d
        if a >= 1 and a in s and (a+d) in s and (a+2*d) in s:
            return True
        # x as 3rd term: a = x-2d
        a = x - 2*d
        if a >= 1 and (a+3*d) <= 512 and a in s and (a+d) in s and (a+3*d) in s:
            return True
        # x as 2nd term: a = x-d
        a = x - d
        if a >= 1 and (a+2*d) <= 512 and (a+3*d) <= 512 and a in s and (a+2*d) in s and (a+3*d) in s:
            return True
        # x as 1st term: a = x
        if (x+d) <= 512 and (x+2*d) <= 512 and (x+3*d) <= 512 and (x+d) in s and (x+2*d) in s and (x+3*d) in s:
            return True
    return False

def construct():
    # Base-3 0/1 construction
    base_set = set()
    for i in range(1, 513):
        # Check if i has only 0/1 digits in base 3
        n = i
        ok = True
        while n > 0:
            if n % 3 == 2:
                ok = False
                break
            n //= 3
        if ok:
            base_set.add(i)
    
    # Try to add more elements
    improved = True
    current = set(base_set)
    while improved:
        improved = False
        candidates = [x for x in range(1, 513) if x not in current]
        random.shuffle(candidates)
        for x in candidates:
            if not has_4ap_with(current, x):
                current.add(x)
                improved = True
                break
    
    # Local search: try to swap
    for _ in range(100):
        # Try removing one element and adding two
        elements = list(current)
        random.shuffle(elements)
        removed = None
        for rem in elements:
            trial = current - {rem}
            added = 0
            for x in range(1, 513):
                if x not in trial and not has_4ap_with(trial, x):
                    trial.add(x)
                    added += 1
                    if added >= 2:
                        break
            if added >= 2:
                current = trial
                break
        if removed is None:
            break
    
    return sorted(current)
