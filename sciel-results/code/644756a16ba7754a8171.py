def construct():
    from math import gcd
    from itertools import combinations
    
    # Greedy construction with verification
    selected = []
    for x in range(1, 129):
        # Check if adding x violates the condition
        valid = True
        for a, b in combinations(selected, 2):
            g_ab = gcd(a, b)
            g_ax = gcd(a, x)
            g_bx = gcd(b, x)
            if g_ab == g_ax == g_bx:
                valid = False
                break
        if valid:
            selected.append(x)
    
    # Try to improve with local search: try adding removed numbers
    # First, let's see which numbers were rejected
    all_nums = set(range(1, 129))
    current_set = set(selected)
    rejected = all_nums - current_set
    
    # Try to add back rejected numbers one by one
    improved = True
    while improved:
        improved = False
        for x in sorted(rejected):
            if x in current_set:
                continue
            valid = True
            for a, b in combinations(current_set, 2):
                g_ab = gcd(a, b)
                g_ax = gcd(a, x)
                g_bx = gcd(b, x)
                if g_ab == g_ax == g_bx:
                    valid = False
                    break
            if valid:
                current_set.add(x)
                rejected.discard(x)
                improved = True
    
    return list(current_set)
