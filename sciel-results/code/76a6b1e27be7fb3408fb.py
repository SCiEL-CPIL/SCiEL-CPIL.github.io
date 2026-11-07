"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
from collections import defaultdict

NMAX = 256

# Pre-compute all conflict relationships
# conflicts[x] = set of pairs (b,c) such that x | b*c or b | x*c or c | x*b for distinct b,c
# Actually, let's precompute: for each x, which other numbers conflict with it in a triple
# conflict_pairs[x] = set of (y,z) where x,y,z are distinct and x|y*z or y|x*z or z|x*y

def _precompute_conflicts():
    """Precompute conflict relationships between numbers."""
    # For each pair (x,y), determine if they can coexist in a valid set
    # Two numbers x,y conflict if there exists z such that {x,y,z} is invalid
    # But that's complex. Instead, let's precompute for each x, the set of y such that
    # x and y are "incompatible" in some sense.
    
    # Actually, the constraint is on triples. Let's precompute:
    # For each x, and each pair (b,c) of distinct elements from 2..256 (b!=c, both != x),
    # check if x|b*c, b|x*c, or c|x*b.
    
    # This is expensive. Let's use a different approach:
    # For the greedy removal heuristic, we need to compute conflict degree.
    # Conflict degree of x = number of triples (x,b,c) or (b,x,c) or (b,c,x) that are invalid.
    
    # Let's precompute a matrix: compat[x][y] = True if x and y can both be in a valid set
    # (i.e., there's no z such that {x,y,z} violates the constraint)
    # This is still expensive. Let's just compute on the fly but with caching.
    
    pass

def _can_add(x, S):
    """
    Check if x can be added to set S.
    S is a list of ints.
    """
    n = len(S)
    for i in range(n):
        b = S[i]
        for j in range(i + 1, n):
            c = S[j]
            if (b * c) % x == 0:
                return False
            if (x * c) % b == 0:
                return False
            if (x * b) % c == 0:
                return False
    return True

def _conflict_degree(x, S):
    """Count how many triples involving x would be invalid if x were in S."""
    count = 0
    n = len(S)
    for i in range(n):
        b = S[i]
        for j in range(i + 1, n):
            c = S[j]
            if (b * c) % x == 0:
                count += 1
            if (x * c) % b == 0:
                count += 1
            if (x * b) % c == 0:
                count += 1
    return count

def _greedy_addition(order, deadline):
    """Greedy: add elements in given order if compatible."""
    S = []
    S_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if _can_add(x, S):
            S.append(x)
            S_set.add(x)
    return S

def _is_valid(S):
    """Check if the set S is valid."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            for k in range(j + 1, n):
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
                if (a * c) % b == 0:
                    return False
                if (a * b) % c == 0:
                    return False
    return True

def _local_search_improve(S, deadline, max_iters=500, rng_seed=42):
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(rng_seed)
    
    # Pre-compute candidates not in S
    candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        progress = False
        
        # 1. Try to add candidates
        if candidates:
            # Try multiple candidates, prioritize those with lower conflict
            trial_count = min(15, len(candidates))
            trial_candidates = rng.sample(candidates, trial_count)
            
            # Sort by conflict degree (lower is better)
            trial_candidates.sort(key=lambda x: _conflict_degree(x, S))
            
            for x in trial_candidates:
                if _can_add(x, S):
                    S.append(x)
                    S_set.add(x)
                    candidates.remove(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    progress = True
                    break
            
        if progress:
            continue
            
        # 2. Try swaps: remove one, add another
        if not S or not candidates:
            break
            
        # Try several removal candidates
        n_try = min(5, len(S))
        for _ in range(n_try):
            idx_remove = rng.randrange(len(S))
            removed = S[idx_remove]
            
            S_temp = S[:]
            S_temp.pop(idx_remove)
            S_temp_set = S_set - {removed}
            
            # Try to find a replacement from candidates
            trial_candidates = rng.sample(candidates, min(25, len(candidates)))
            trial_candidates.sort(key=lambda x: _conflict_degree(x, S_temp))
            
            swapped = False
            for x in trial_candidates:
                if _can_add(x, S_temp):
                    S = S_temp + [x]
                    S_set = S_temp_set | {x}
                    if removed not in candidates:
                        candidates.append(removed)
                    if x in candidates:
                        candidates.remove(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    swapped = True
                    progress = True
                    break
            
            if swapped:
                break
        
        if not progress and not candidates:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin

    best_S = None
    best_size = 0
    
    # Phase 1: Diverse Greedy Initializations
    # 1. Largest First
    order_largest = list(range(NMAX, 1, -1))
    S1 = _greedy_addition(order_largest, deadline)
    if S1 and _is_valid(S1) and len(S1) > best_size:
        best_S = S1
        best_size = len(S1)
        
    # 2. Smallest First
    order_smallest = list(range(2, NMAX + 1))
    S2 = _greedy_addition(order_smallest, deadline)
    if S2 and _is_valid(S2) and len(S2) > best_size:
        best_S = S2
        best_size = len(S2)
    
    # 4. Randomized Orders
    for seed in [1, 7, 42, 100, 200, 300, 400]:
        if time.time() > deadline - 25:
            break
        rng = random.Random(seed)
        order_rand = list(range(2, NMAX + 1))
        rng.shuffle(order_rand)
        Sr = _greedy_addition(order_rand, deadline)
        if Sr and _is_valid(Sr) and len(Sr) > best_size:
            best_S = Sr
            best_size = len(Sr)
            
    if not best_S:
        best_S = [2, 3] 
        best_size = 2

    # Phase 2: Local Search Improvement with multiple runs
    # Run 1 on best_S
    if time.time() < deadline - 5:
        S_improved = _local_search_improve(best_S, deadline - 5, max_iters=500, rng_seed=42)
        if S_improved and _is_valid(S_improved) and len(S_improved) > best_size:
            best_S = S_improved
            best_size = len(S_improved)
            
    # Run 2 on perturbed version
    if time.time() < deadline - 2:
        rng_perturb = random.Random(123)
        S_perturbed = best_S[:]
        for _ in range(rng_perturb.randint(1, 2)):
            if S_perturbed:
                idx = rng_perturb.randrange(len(S_perturbed))
                S_perturbed.pop(idx)
                
        S_improved2 = _local_search_improve(S_perturbed, deadline - 2, max_iters=300, rng_seed=123)
        if S_improved2 and _is_valid(S_improved2) and len(S_improved2) > best_size:
            best_S = S_improved2
            best_size = len(S_improved2)

    # Final Validation
    if not _is_valid(best_S):
        # Fallback to a known valid set if something went wrong
        # Primes are always valid
        primes = [i for i in range(2, NMAX + 1) if all(i % j != 0 for j in range(2, int(i**0.5) + 1))]
        if primes:
            best_S = primes
        else:
            best_S = [2, 3]

    return [int(x) for x in best_S]
