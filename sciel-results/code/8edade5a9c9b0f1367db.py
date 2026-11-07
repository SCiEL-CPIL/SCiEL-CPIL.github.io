"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def _all_subset_sums(S):
    """Compute all nonempty subset sums of S using iterative doubling."""
    n = len(S)
    if n == 0:
        return set()
    sums = {0}
    for x in S:
        sums = sums | {s + x for s in sums}
    sums.discard(0)
    return sums

def _check_divisibility(sums, total):
    """Check if any two distinct sums have one dividing the other."""
    sums_list = sorted(sums)
    for s in sums_list:
        if s == 0:
            continue
        multiple = 2 * s
        while multiple <= total:
            if multiple in sums:
                return False
            multiple += s
    return True

def _is_valid(S):
    """Return True iff S satisfies the constraint."""
    if not S:
        return True
    total = sum(S)
    sums = _all_subset_sums(S)
    return _check_divisibility(sums, total)

def _fast_valid_check(S):
    """Fast validity check using iterative subset sums."""
    if not S:
        return True
    total = sum(S)
    sums = _all_subset_sums(S)
    return _check_divisibility(sums, total)

def _try_swap(S, remove_idx, add_val, pool_set):
    """Try swapping S[remove_idx] with add_val. Returns (new_S, True) if valid."""
    new_S = S[:remove_idx] + S[remove_idx+1:] + [add_val]
    # Check distinctness
    if len(set(new_S)) != len(new_S):
        return None, False
    # Check range
    if any(x < 1 or x > 256 for x in new_S):
        return None, False
    # Check validity
    if _fast_valid_check(new_S):
        return new_S, True
    return None, False

def _simulated_annealing(initial_S, pool, start_time, deadline):
    """Run simulated annealing to find a larger valid set."""
    rng = random.Random(42)
    current = list(initial_S)
    current_sums = _all_subset_sums(current)
    best = list(current)
    best_size = len(current)
    
    used = set(current)
    available = [x for x in pool if x not in used]
    
    # Temperature schedule: start high, cool down
    T_start = 5.0
    T_end = 0.1
    total_steps = 20000
    
    step = 0
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and step < total_steps:
        step += 1
        T = T_start * (T_end / T_start) ** (step / total_steps)
        
        if not available:
            break
        
        # Pick a random element to remove
        rem_idx = rng.randint(0, len(current) - 1)
        rem_val = current[rem_idx]
        
        # Pick a random element to add
        add_idx = rng.randint(0, len(available) - 1)
        add_val = available[add_idx]
        
        # Try the swap
        new_S, valid = _try_swap(current, rem_idx, add_val, used)
        
        if valid:
            # Accept valid move
            current = new_S
            used.discard(rem_val)
            used.add(add_val)
            available.remove(add_val)
            available.append(rem_val)
            
            if len(current) > best_size:
                best = list(current)
                best_size = len(current)
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > max_no_improve and len(current) == best_size:
                # Reheat occasionally
                T = max(T, T_start * 0.5)
                no_improve = 0
        else:
            # Invalid move: reject
            no_improve += 1
    
    return best

def _greedy_extend(S, pool, deadline):
    """Try to greedily add elements to S."""
    current = list(S)
    used = set(current)
    available = [x for x in pool if x not in used]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(available)
        for x in available:
            if time.time() > deadline:
                break
            test = current + [x]
            if len(set(test)) == len(test) and _fast_valid_check(test):
                current.append(x)
                used.add(x)
                available.remove(x)
                improved = True
                break
    
    return current

def _random_initial_set(pool, size, rng):
    """Generate a random valid set of given size."""
    # Start with large primes (less likely to create divisibility)
    primes = [x for x in range(2, 257) if all(x % i != 0 for i in range(2, int(math.isqrt(x)) + 1))]
    rng.shuffle(primes)
    
    S = []
    used = set()
    
    # Add primes first
    for p in primes:
        if len(S) >= size:
            break
        test = S + [p]
        if len(set(test)) == len(test) and _fast_valid_check(test):
            S.append(p)
            used.add(p)
    
    # Fill with random numbers
    remaining = [x for x in pool if x not in used]
    rng.shuffle(remaining)
    for x in remaining:
        if len(S) >= size:
            break
        test = S + [x]
        if len(set(test)) == len(test) and _fast_valid_check(test):
            S.append(x)
            used.add(x)
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    pool = list(range(1, 257))
    
    best_set = []
    best_size = 0
    
    # Phase 1: Generate diverse initial sets via greedy with different orderings
    rng = random.Random(12345)
    
    # Strategy A: Primes first
    primes = [x for x in range(2, 257) if all(x % i != 0 for i in range(2, int(math.isqrt(x)) + 1))]
    rng.shuffle(primes)
    non_primes = [x for x in pool if x not in set(primes)]
    rng.shuffle(non_primes)
    ordered = primes + non_primes
    
    S = []
    for x in ordered:
        test = S + [x]
        if _fast_valid_check(test):
            S.append(x)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Strategy B: Large numbers first (harder to divide)
    ordered = sorted(pool, reverse=True)
    rng.shuffle(ordered)
    S = []
    for x in ordered:
        test = S + [x]
        if _fast_valid_check(test):
            S.append(x)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Strategy C: Random
    for attempt in range(5):
        if time.time() > deadline - 5:
            break
        ordered = pool[:]
        rng.shuffle(ordered)
        S = []
        for x in ordered:
            test = S + [x]
            if _fast_valid_check(test):
                S.append(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    # Phase 2: Simulated Annealing from best
    if best_set and time.time() < deadline - 5:
        improved = _simulated_annealing(best_set, pool, start_time, deadline)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(improved)
    
    # Phase 3: Greedy extend from best
    if best_set and time.time() < deadline - 2:
        extended = _greedy_extend(best_set, pool, deadline)
        if len(extended) > best_size:
            best_set = extended
            best_size = len(extended)
    
    # Phase 4: Another SA round if time permits
    if best_set and time.time() < deadline - 3:
        improved = _simulated_annealing(best_set, pool, start_time, deadline)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(improved)
    
    # Final verification
    if best_set and not _is_valid(best_set):
        best_set = []
    
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
