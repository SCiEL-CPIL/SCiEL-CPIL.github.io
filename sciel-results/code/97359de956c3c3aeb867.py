import time
import random

def _can_add(x, selected):
    """
    Check if x can be added to selected.
    Constraints:
    1. x does not divide sum of any 3 distinct elements in selected.
    2. No a in selected divides sum of any 3 distinct elements in (selected U {x}) \ {a}.
       This is equivalent to: No a in selected divides (x + b + c) for distinct b, c in selected \ {a}.
    """
    m = len(selected)
    if m == 0:
        return True
    
    # 1. Check if x divides any 3-sum from selected
    # We can just iterate all triplets. m <= 48, so 48C3 ~ 17k checks.
    for i in range(m):
        for j in range(i+1, m):
            for k in range(j+1, m):
                if (selected[i] + selected[j] + selected[k]) % x == 0:
                    return False

    # 2. Check if any a in selected divides x + b + c
    # We iterate over pairs (b, c) in selected.
    # For each pair, sum_bc = b + c.
    # We need to check if (x + sum_bc) % a == 0 for any a in selected such that a != b and a != c.
    
    # Optimization: Precompute sums of pairs? 
    # Since m is small, direct iteration is fine.
    for i in range(m):
        for j in range(i+1, m):
            s_pair = selected[i] + selected[j]
            target = x + s_pair
            # Check against all a in selected except i and j
            for k in range(m):
                if k == i or k == j:
                    continue
                if target % selected[k] == 0:
                    return False
    return True

def _greedy_build(order, time_limit):
    """Greedy build following given order with time limit."""
    selected = []
    start_time = time.time()
    for x in order:
        if len(selected) >= 48:
            break
        if time.time() - start_time > time_limit:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, time_limit):
    """
    Local search: Try to add any missing number. If none, try swapping one out for another.
    """
    start_time = time.time()
    best = list(selected)
    best_set = set(best)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try to add any candidate
        candidates = [x for x in range(1, 193) if x not in best_set]
        # Sort by "safety" - larger numbers are often safer as they are less likely to divide sums
        # But also, numbers with fewer factors are safer.
        # Let's just try descending order first as it was effective.
        candidates.sort(reverse=True)
        
        added = False
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            if _can_add(x, best):
                best.append(x)
                best_set.add(x)
                improved = True
                added = True
                break
        
        if added:
            continue
            
        # 2. If no addition, try swaps
        # Try removing each element and seeing if we can add a different one
        # To speed up, we only try a few swaps per iteration
        
        # Shuffle indices to avoid bias
        indices = list(range(len(best)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            
            removed = best[idx]
            temp_best = best[:idx] + best[idx+1:]
            temp_set = set(temp_best)
            
            # Try to add a candidate not in temp_set
            # We prefer candidates larger than removed or just any valid one
            candidates_for_swap = [x for x in range(1, 193) if x not in temp_set]
            # Heuristic: Try larger numbers first
            candidates_for_swap.sort(reverse=True)
            
            swapped = False
            for x in candidates_for_swap:
                if time.time() - start_time > time_limit:
                    break
                if x == removed:
                    continue
                if _can_add(x, temp_best):
                    best = temp_best + [x]
                    best_set = set(best)
                    improved = True
                    swapped = True
                    break
            
            if swapped:
                break
                
        if not improved:
            # Try a few random restarts of local search or break
            # To avoid infinite loop if stuck, we break after no improvement
            # But we can try shuffling the order of checks
            if random.random() > 0.5:
                break
            # else continue to try different random orders in next loop iteration?
            # Actually, if no improvement in a full pass, we are likely at a local optimum.
            # Let's break to save time for other strategies.
            break
            
    return best

def construct():
    start_time = time.time()
    # Hard deadline: 50 seconds. Evaluator kills at 60s.
    deadline = start_time + 50.0
    
    best_result = []
    
    # Strategy 1: Descending Greedy
    order1 = list(range(192, 0, -1))
    r1 = _greedy_build(order1, 5.0)
    if len(r1) > len(best_result):
        best_result = r1
    
    # Strategy 2: Randomized Greedies
    for seed in range(100):
        if time.time() > deadline - 10.0:
            break
        
        random.seed(seed)
        
        if seed % 4 == 0:
            # Descending with noise
            order = list(range(192, 0, -1))
            random.shuffle(order)
            # Re-sort to keep large numbers generally first but perturbed
            order.sort(key=lambda x: x + random.uniform(-20, 20), reverse=True)
        elif seed % 4 == 1:
            # Pure Random
            order = list(range(1, 193))
            random.shuffle(order)
        elif seed % 4 == 2:
            # Primality bias (primes are good divisors? No, primes are good to NOT divide sums?
            # Actually, if x is prime, x divides S only if S is multiple of x.
            # Small primes are bad because they divide many sums. Large primes are safer.)
            def is_prime(n):
                if n < 2: return False
                for i in range(2, int(n**0.5)+1):
                    if n % i == 0: return False
                return True
            order = list(range(1, 193))
            random.shuffle(order)
            order.sort(key=lambda x: (is_prime(x), x), reverse=True)
        else:
            # Medium numbers first
            order = list(range(1, 193))
            random.shuffle(order)
            order.sort(key=lambda x: abs(x - 100))
            
        r = _greedy_build(order, 2.0)
        
        if len(r) > len(best_result):
            best_result = r
            # If we found a new best, do a quick local search
            if time.time() < deadline - 5.0:
                r_ls = _local_search(r, 2.0)
                if len(r_ls) > len(best_result):
                    best_result = r_ls
        elif len(r) == len(best_result) and len(r) > 0:
            # Even if same size, maybe local search improves? Unlikely if greedy is strong.
            pass

    # Strategy 3: Final Local Search on the best result found
    if time.time() < deadline - 2.0:
        remaining = deadline - time.time() - 1.0
        if remaining > 1.0:
            best_result = _local_search(best_result, remaining)

    return best_result
