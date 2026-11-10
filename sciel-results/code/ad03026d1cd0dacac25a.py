import time

def construct():
    start_time = time.time()
    limit = 180
    # Greedy construction from largest to smallest
    S = []
    S_set = set()
    
    for a in range(limit, 0, -1):
        conflict = False
        for b in S:
            if b <= a:
                continue
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if c != a and c != b and c in S_set:
                    conflict = True
                    break
        if not conflict:
            S.append(a)
            S_set.add(a)
        
        if time.time() - start_time > 50:
            break
    
    # Local search: try to add back rejected numbers by removing conflicting partners
    # This can sometimes increase total size
    current_size = len(S)
    S_list = list(S)
    S_set2 = set(S_list)
    
    # For each number not in S, check if we can add it by removing minimal conflicts
    # Strategy: for each x not in S, find conflicts with S
    # If removing one element from S allows adding x and net gain >= 0, do it
    # But we want net gain > 0, so we need to remove at most 0 elements (impossible) 
    # or remove 1 element and add 1 (net 0, no gain)
    # Actually, adding x back might allow other numbers too... complex.
    # 
    # Simpler improvement: try different greedy orders
    # Since time is available, let's try a few different orderings
    
    best_size = len(S_list)
    best_set = S_set2.copy()
    
    # Try: greedy from smallest to largest
    S2 = []
    S2_set = set()
    for a in range(1, limit + 1):
        conflict = False
        for b in S2:
            if b < a:
                # a is the larger, so a = b or a = c
                # 1/a = 1/b + 1/c with a > b means c > a is impossible since 1/c < 1/a - 1/b
                # Actually if a > b, then 1/a < 1/b, so 1/c = 1/a - 1/b < 0. Impossible.
                # So only need to check when the new element could be the smaller one
                continue
            # b >= a, so b could be the larger element
            # 1/a = 1/b + 1/c => c = ab/(b-a)
            if b == a:
                continue
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if c != a and c != b and c in S2_set:
                    conflict = True
                    break
        if not conflict:
            S2.append(a)
            S2_set.add(a)
    
    if len(S2) > best_size:
        best_size = len(S2)
        best_set = S2_set.copy()
    
    # Try: greedy with random perturbations of order
    import random
    rng = random.Random(42)
    
    for trial in range(100):
        if time.time() - start_time > 55:
            break
        
        # Create a perturbed order: mostly descending but with some shuffling
        order = list(range(limit, 0, -1))
        # Swap some adjacent elements
        for _ in range(5):
            i = rng.randint(0, limit - 2)
            order[i], order[i+1] = order[i+1], order[i]
        
        S3 = []
        S3_set = set()
        for a in order:
            conflict = False
            for b in S3:
                if b <= a:
                    continue
                num = a * b
                den = b - a
                if num % den == 0:
                    c = num // den
                    if c != a and c != b and c in S3_set:
                        conflict = True
                        break
            if not conflict:
                S3.append(a)
                S3_set.add(a)
        
        if len(S3) > best_size:
            best_size = len(S3)
            best_set = S3_set.copy()
    
    return list(best_set)

