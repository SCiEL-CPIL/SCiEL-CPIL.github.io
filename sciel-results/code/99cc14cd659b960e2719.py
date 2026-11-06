import time
import random

def construct():
    start = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill

    N = 100
    target = 200

    # Helper: compute reachable sums up to target for a given set S using boolean array
    def compute_reachable_arr(S, target=target):
        reachable = [False] * (target + 1)
        reachable[0] = True
        for x in S:
            for s in range(target - x, -1, -1):
                if reachable[s]:
                    reachable[s + x] = True
        return reachable

    # Helper: check if adding x to a set with current reachable sums would create target
    def would_create_target(x, reachable, target=target):
        return reachable[target - x]

    # Helper: add x to reachable array
    def add_to_reachable(x, reachable, target=target):
        for s in range(target - x, -1, -1):
            if reachable[s]:
                reachable[s + x] = True

    # Step 1: Start with structural seed {67, ..., 100}
    # This set is provably valid:
    # - 1-element: all in [67,100], none equal 200
    # - 2-element: max pair 99+100=199 < 200
    # - 3-element: min triple 67+68+69=204 > 200
    S = set(range(67, 101))  # 34 elements
    reachable = compute_reachable_arr(S)

    # Step 2: Greedy addition from 66 down to 1 (larger first, less likely to create small sums)
    for x in range(66, 0, -1):
        if time.time() - start > DEADLINE:
            break
        if not would_create_target(x, reachable):
            S.add(x)
            add_to_reachable(x, reachable)

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
        reachable_try = compute_reachable_arr(S_try)
        for x in ordering:
            if x in S_try:
                continue
            if not would_create_target(x, reachable_try):
                S_try.add(x)
                add_to_reachable(x, reachable_try)
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try

    # Step 4: Randomized restarts with different starting seeds and orderings
    random.seed(42)
    for _ in range(50):
        if time.time() - start > DEADLINE:
            break
        
        # Try different structural seeds
        seed_start = random.randint(55, 75)
        S_try = set(range(seed_start, 101))
        
        # Verify the seed is valid
        reachable_try = compute_reachable_arr(S_try)
        if reachable_try[target]:
            # Remove elements that cause violations
            for x in sorted(S_try, reverse=True):
                if reachable_try[target]:
                    S_test = S_try - {x}
                    reachable_test = compute_reachable_arr(S_test)
                    if not reachable_test[target]:
                        S_try = S_test
                        reachable_try = reachable_test
                    else:
                        S_try.discard(x)
                        reachable_try = compute_reachable_arr(S_try)
                else:
                    break
        
        if reachable_try[target]:
            continue
            
        # Greedy addition
        rem = [x for x in range(1, 101) if x not in S_try]
        random.shuffle(rem)
        
        for x in rem:
            if time.time() - start > DEADLINE:
                break
            if not would_create_target(x, reachable_try):
                S_try.add(x)
                add_to_reachable(x, reachable_try)
        
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try

    # Step 5: Local search - try to swap elements to increase cardinality
    if time.time() - start < DEADLINE:
        S_work = set(best_set)
        reachable_work = compute_reachable_arr(S_work)
        
        # Try multiple passes of local improvement
        for _ in range(30):
            if time.time() - start > DEADLINE:
                break
            
            improved = False
            # Try each element in the set for removal
            for x in list(S_work):
                if time.time() - start > DEADLINE:
                    break
                
                # Remove x and compute what sums are no longer reachable
                S_test = S_work - {x}
                reachable_test = compute_reachable_arr(S_test)
                
                # Try to add elements not in S_test
                added = []
                for y in range(100, 0, -1):
                    if y in S_test:
                        continue
                    if not would_create_target(y, reachable_test):
                        S_test.add(y)
                        add_to_reachable(y, reachable_test)
                        added.append(y)
                    
                    # Stop if we've added 2 elements (net gain of 1)
                    if len(added) >= 2:
                        break
                
                if len(added) >= 2:
                    S_work = S_test
                    reachable_work = reachable_test
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
