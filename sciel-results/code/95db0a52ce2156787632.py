import time
import random

def construct():
    start = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill

    N = 100
    target = 200

    # Helper: compute reachable sums up to target for a given set S
    def compute_reachable(S, target=target):
        reachable = {0}
        for x in S:
            new_sums = {s + x for s in reachable if s + x <= target}
            reachable |= new_sums
        return reachable

    # Helper: check if adding x to a set with current reachable sums would create target
    def would_create_target(x, reachable, target=target):
        return (target - x) in reachable

    # Step 1: Start with structural seed {67, ..., 100}
    # This set is provably valid:
    # - 1-element: all in [67,100], none equal 200
    # - 2-element: max pair 99+100=199 < 200
    # - 3-element: min triple 67+68+69=204 > 200
    S = set(range(67, 101))  # 34 elements
    reachable = compute_reachable(S)

    # Step 2: Greedy addition from 66 down to 1 (larger first, less likely to create small sums)
    for x in range(66, 0, -1):
        if time.time() - start > DEADLINE:
            break
        if not would_create_target(x, reachable):
            S.add(x)
            reachable |= {s + x for s in reachable if s + x <= target}

    best_set = set(S)
    best_size = len(S)

    # Step 3: Try different orderings for greedy addition
    remaining = [x for x in range(1, 101) if x not in S]

    orderings = [
        remaining,
        sorted(remaining, reverse=True),
        sorted(remaining),
    ]

    for ordering in orderings:
        if time.time() - start > DEADLINE:
            break
        S_try = set(S)
        reachable_try = compute_reachable(S_try)
        for x in ordering:
            if x in S_try:
                continue
            if not would_create_target(x, reachable_try):
                S_try.add(x)
                reachable_try |= {s + x for s in reachable_try if s + x <= target}
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try

    # Step 4: Randomized restarts with different starting seeds and orderings
    random.seed(42)
    for _ in range(20):
        if time.time() - start > DEADLINE:
            break
        
        # Try different structural seeds
        seed_start = random.randint(60, 75)
        S_try = set(range(seed_start, 101))
        
        # Verify the seed is valid
        reachable_try = compute_reachable(S_try)
        if target in reachable_try:
            # Remove elements that cause violations
            for x in sorted(S_try, reverse=True):
                if target in reachable_try:
                    # Recompute without x
                    S_test = S_try - {x}
                    reachable_test = compute_reachable(S_test)
                    if target not in reachable_test:
                        S_try = S_test
                        reachable_try = reachable_test
                    else:
                        S_try.discard(x)
                        reachable_try = compute_reachable(S_try)
                else:
                    break
        
        if target in reachable_try:
            continue
            
        # Greedy addition
        remaining = [x for x in range(1, 101) if x not in S_try]
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() - start > DEADLINE:
                break
            if not would_create_target(x, reachable_try):
                S_try.add(x)
                reachable_try |= {s + x for s in reachable_try if s + x <= target}
        
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try

    # Step 5: Local search - try to swap elements to increase cardinality
    # Try removing one element and adding two others
    if time.time() - start < DEADLINE:
        S_work = set(best_set)
        reachable_work = compute_reachable(S_work)
        in_set = set(S_work)
        
        for _ in range(5):
            if time.time() - start > DEADLINE:
                break
            
            improved = False
            # Try each element in the set
            for x in list(S_work):
                if time.time() - start > DEADLINE:
                    break
                
                S_test = S_work - {x}
                reachable_test = compute_reachable(S_test)
                
                # Try to add elements not in S_test
                for y in range(100, 0, -1):
                    if y in S_test:
                        continue
                    if not would_create_target(y, reachable_test):
                        S_test.add(y)
                        reachable_test |= {s + y for s in reachable_test if s + y <= target}
                
                # Try to add one more
                for z in range(100, 0, -1):
                    if z in S_test:
                        continue
                    if not would_create_target(z, reachable_test):
                        S_test.add(z)
                        reachable_test |= {s + z for s in reachable_test if s + z <= target}
                        break
                
                if len(S_test) > len(S_work):
                    S_work = S_test
                    reachable_work = reachable_test
                    in_set = set(S_work)
                    best_set = S_work
                    best_size = len(S_work)
                    improved = True
                    break
            
            if not improved:
                break

    result = list(best_set)
    if not result:
        result = [100]  # fallback, always valid
    
    return result
