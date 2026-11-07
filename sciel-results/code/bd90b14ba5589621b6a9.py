"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 40, 'maximum_size': 18}
START_TIME = time.time()
TIME_LIMIT = 85  # seconds

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    candidates = list(range(2, P['n'] + 1))
    
    def complexity(x):
        n = x
        count = 0
        exp_sum = 0
        for p in primes:
            while n % p == 0:
                count += 1
                exp_sum += 1
                n //= p
        return (count, exp_sum)
        
    candidates.sort(key=complexity)
    
    # Greedy pass
    greedy_set = []
    greedy_products = {1}
    for x in candidates:
        new_products = {p * x for p in greedy_products}
        if not (new_products & greedy_products):
            greedy_set.append(x)
            greedy_products |= new_products
            
    best_set = greedy_set[:]
    best_size = len(best_set)
    
    # Backtracking to improve
    def search(current_set, current_products, remaining_candidates, target_size):
        nonlocal best_set, best_size
        
        if time.time() - START_TIME > TIME_LIMIT:
            return
        
        if len(current_set) >= target_size:
            if len(current_set) > best_size:
                best_set = current_set[:]
                best_size = len(best_set)
            return
        
        if not remaining_candidates:
            if len(current_set) > best_size:
                best_set = current_set[:]
                best_size = len(best_set)
            return
        
        # Pruning: even if we add all remaining, can we beat best_size?
        if len(current_set) + len(remaining_candidates) <= best_size:
            return
            
        for i, x in enumerate(remaining_candidates):
            new_products = {p * x for p in current_products}
            if new_products & current_products:
                continue
            
            new_set = current_set + [x]
            new_products_set = current_products | new_products
            new_remaining = remaining_candidates[i+1:]
            
            search(new_set, new_products_set, new_remaining, target_size)
            
            if len(best_set) >= target_size:
                return

    # Try to find larger sets
    # Start from best_size + 1
    for target in range(best_size + 1, P['maximum_size'] + 1):
        search([], {1}, candidates, target)
        if best_size >= target:
            continue
        else:
            # If we couldn't find target, maybe target-1 was the max?
            # But we already have best_size.
            # If search for target failed, we stop?
            # Not necessarily, but likely.
            # Given time constraints, we stop if we fail to find target.
            break
            
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
