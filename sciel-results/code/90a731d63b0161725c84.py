"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _is_valid(S):
    """Return True iff S (list of distinct ints in 1..256) satisfies the constraint:
    no two distinct nonempty subset sums s, t with s != t have s | t or t | s.
    """
    if not S:
        return True
    total = sum(S)
    n = len(S)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += S[i]
        sums.add(s)
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

def _subset_sums(S):
    """Compute all nonempty subset sums of S."""
    n = len(S)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += S[i]
        sums.add(s)
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

def _incremental_check(S, new_elem):
    """Check if adding new_elem to S preserves the divisibility-free property.
    Uses incremental subset sum update: new sums = old_sums ∪ {s + new_elem for s in old_sums ∪ {0}}.
    Returns (True, new_sums) if valid, (False, None) otherwise.
    """
    n = len(S)
    if n > 20:
        # Too large for bitmask; do full check
        test = S + [new_elem]
        return _is_valid(test), _subset_sums(test) if _is_valid(test) else None
    
    # Compute old subset sums
    old_sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += S[i]
        old_sums.add(s)
    
    # New sums: old_sums plus (old_sums + new_elem) plus {new_elem}
    new_sums = set(old_sums)
    for s in old_sums:
        new_sums.add(s + new_elem)
    new_sums.add(new_elem)
    
    total = sum(S) + new_elem
    if not _check_divisibility(new_sums, total):
        return False, None
    return True, new_sums

def _candidate_score(x, current_sums, used_set):
    """Score a candidate: prefer numbers that create fewer divisibility conflicts.
    Lower score = better candidate."""
    # Count how many existing sums divide x or are divided by x
    conflict_count = 0
    for s in current_sums:
        if s == 0:
            continue
        if x % s == 0 or s % x == 0:
            conflict_count += 1
    # Also penalize if x has many divisors/multiples in 1..256 that could be in pool
    div_count = sum(1 for d in range(1, x+1) if x % d == 0)
    mult_count = sum(1 for m in range(2*x, 257, x))
    return conflict_count * 10 + div_count + mult_count

def _greedy_build(pool, deadline, seed=None):
    """Build a set greedily using candidate scoring."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    current = []
    current_sums = set()
    used = set()
    available = pool[:]
    
    while time.time() < deadline:
        if not available:
            break
        
        # Score all available candidates
        scored = []
        for x in available:
            sc = _candidate_score(x, current_sums, used)
            scored.append((sc, x))
        
        # Sort by score, add some randomness
        scored.sort(key=lambda t: t[0])
        
        # Try top candidates with some randomization
        tried = 0
        max_tries = min(10, len(scored))
        indices = list(range(max_tries))
        rng.shuffle(indices)
        
        added = False
        for idx in indices:
            x = scored[idx][1]
            valid, new_sums = _incremental_check(current, x)
            if valid:
                current.append(x)
                current_sums = new_sums
                used.add(x)
                available.remove(x)
                added = True
                break
            tried += 1
        
        if not added:
            # No more elements can be added
            break
    
    return current

def _local_search(S, pool, deadline):
    """Local search: try add, swap (remove one add one), and 2-for-1 swap."""
    current = list(S)
    used = set(current)
    available = [x for x in pool if x not in used]
    current_sums = _subset_sums(current) if current else set()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding elements
        if available:
            random.shuffle(available)
            for x in available:
                if time.time() > deadline:
                    break
                valid, new_sums = _incremental_check(current, x)
                if valid:
                    current.append(x)
                    current_sums = new_sums
                    used.add(x)
                    available.remove(x)
                    improved = True
        
        # Try 1-for-1 swap
        if not improved and len(current) > 1 and available:
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                temp = current[:i] + current[i+1:]
                temp_sums = _subset_sums(temp) if temp else set()
                
                # Try to find a replacement
                for x in available[:30]:
                    if time.time() > deadline:
                        break
                    valid, new_sums = _incremental_check(temp, x)
                    if valid:
                        current = temp + [x]
                        current_sums = new_sums
                        used.discard(removed)
                        used.add(x)
                        available.append(removed)
                        available.remove(x)
                        improved = True
                        break
                if improved:
                    break
        
        # Try 2-for-1 swap: remove two elements, add one (can only help if we're stuck)
        # Actually this reduces size, so skip. Instead try 1-for-2: remove one, add two
        if not improved and len(current) >= 1 and len(available) >= 2:
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                temp = current[:i] + current[i+1:]
                temp_sums = _subset_sums(temp) if temp else set()
                
                # Try to add two elements from available
                avail_subset = available[:20]
                random.shuffle(avail_subset)
                found = False
                for a in avail_subset:
                    if time.time() > deadline:
                        break
                    valid1, sums1 = _incremental_check(temp, a)
                    if valid1:
                        temp2 = temp + [a]
                        for b in avail_subset:
                            if b == a:
                                continue
                            if time.time() > deadline:
                                break
                            valid2, sums2 = _incremental_check(temp2, b)
                            if valid2:
                                current = temp2 + [b]
                                current_sums = sums2
                                used.discard(removed)
                                used.add(a)
                                used.add(b)
                                available.append(removed)
                                available.remove(a)
                                available.remove(b)
                                improved = True
                                found = True
                                break
                    if found:
                        break
                if improved:
                    break
    
    return current

def _divisibility_free_score(x):
    """Score how 'divisibility-free' a number is: fewer divisors and multiples in 1..256."""
    div_count = sum(1 for d in range(1, x+1) if x % d == 0)
    mult_count = sum(1 for m in range(2*x, 257, x))
    return div_count + mult_count

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin before 60s limit
    
    pool = list(range(1, 257))
    
    # Pre-compute divisibility-free scores for all numbers
    dfs = {x: _divisibility_free_score(x) for x in pool}
    
    best_set = []
    best_size = 0
    
    # Phase 1: Multiple greedy builds with different orderings
    # Strategy A: Sort by divisibility-free score (prefer numbers with few divisors/multiples)
    for attempt in range(3):
        if time.time() > deadline:
            break
        ordered_pool = sorted(pool, key=lambda x: dfs[x])
        # Add randomness
        for i in range(len(ordered_pool) - 1, 0, -1):
            j = random.randint(0, i)
            ordered_pool[i], ordered_pool[j] = ordered_pool[j], ordered_pool[i]
        
        S = _greedy_build(ordered_pool, deadline, seed=attempt * 100)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    # Strategy B: Start with large primes (hard to divide)
    primes = [x for x in range(2, 257) if all(x % i != 0 for i in range(2, int(math.isqrt(x)) + 1))]
    for attempt in range(3):
        if time.time() > deadline:
            break
        random.shuffle(primes)
        non_primes = [x for x in pool if x not in set(primes)]
        random.shuffle(non_primes)
        ordered_pool = primes + non_primes
        S = _greedy_build(ordered_pool, deadline, seed=1000 + attempt * 100)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    # Strategy C: Random restarts
    for attempt in range(10):
        if time.time() > deadline:
            break
        random_pool = pool[:]
        random.shuffle(random_pool)
        S = _greedy_build(random_pool, deadline, seed=5000 + attempt * 100)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    # Phase 2: Local search from best
    if best_set and time.time() < deadline - 3:
        improved = _local_search(best_set, pool, deadline)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(improved)
    
    # Phase 3: More local search iterations if time permits
    if best_set and time.time() < deadline - 2:
        improved = _local_search(best_set, pool, deadline)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(improved)
    
    # Final verification
    if best_set and not _is_valid(best_set):
        # Should not happen, but safety check
        best_set = []
    
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
