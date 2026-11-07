"""Rewritten program: randomized local search with incremental validity and divisibility pre-filter."""
import random
import time
from collections import Counter

# EVOLVE-BLOCK-START

def _incremental_check_add(nums, x, existing_sums):
    """Check if adding x to nums keeps validity.
    existing_sums: set of all nonempty subset sums of nums.
    Returns (new_sums_set, valid) where new_sums_set includes sums with x.
    """
    # New sums when adding x: {x} ∪ {s + x for s in existing_sums}
    # We need to check:
    # 1. No new sum equals an existing sum (or another new sum)
    # 2. No new sum divides or is divisible by any existing sum
    # 3. No two new sums have divisibility (but new sums are all s+x or x, 
    #    and if s1+x divides s2+x or vice versa, need to check)
    
    new_sums = set()
    new_sums.add(x)
    for s in existing_sums:
        new_sums.add(s + x)
    
    # Check for duplicates between new_sums and existing_sums
    if new_sums & existing_sums:
        return None, False
    
    # Check divisibility: for each pair (a, b) where a in existing_sums, b in new_sums,
    # check if a divides b or b divides a
    # Also check within new_sums: for a, b in new_sums with a < b, check b % a == 0
    # And within existing_sums we already know it's valid.
    
    # Efficient check: for each b in new_sums, check against all a in existing_sums
    for b in new_sums:
        for a in existing_sums:
            if a == b:
                return None, False
            if a < b:
                if b % a == 0:
                    return None, False
            else:
                if a % b == 0:
                    return None, False
    
    # Check within new_sums
    new_list = sorted(new_sums)
    for i in range(len(new_list)):
        a = new_list[i]
        for j in range(i + 1, len(new_list)):
            b = new_list[j]
            if b % a == 0:
                return None, False
    
    # Build the full new sums set
    full_sums = existing_sums | new_sums
    return full_sums, True

def _is_valid_full(nums):
    """Full validation: compute all subset sums and check divisibility-free."""
    sums = [0]
    for x in nums:
        sums += [s + x for s in sums]
    sums = sums[1:]
    cnt = Counter(sums)
    for v, c in cnt.items():
        if c >= 2:
            return False
    vals = sorted(cnt.keys())
    n = len(vals)
    for i in range(n):
        a = vals[i]
        if a <= 1:
            if n > 1:
                return False
            continue
        for j in range(i + 1, n):
            b = vals[j]
            if b % a == 0:
                return False
    return True

def _compute_sums(nums):
    """Compute all nonempty subset sums as a set."""
    sums = set()
    current = {0}
    for x in nums:
        new = set()
        for s in current:
            ns = s + x
            new.add(ns)
            sums.add(ns)
        current = current | new
    return sums

def _greedy_build_incremental(rng, pool, max_size=30, deadline=None):
    """Greedy build using incremental validity check."""
    chosen = []
    existing_sums = set()
    available = list(pool)
    while len(chosen) < max_size:
        if deadline is not None and time.time() > deadline:
            break
        best = None
        best_sums = None
        # Try candidates in random order, pick first valid
        shuffled = available[:]
        rng.shuffle(shuffled)
        for x in shuffled:
            new_sums, valid = _incremental_check_add(chosen, x, existing_sums)
            if valid:
                best = x
                best_sums = new_sums
                break
        if best is None:
            break
        chosen.append(best)
        existing_sums = best_sums
        available.remove(best)
    return chosen

def _local_search_incremental(start, rng, deadline, max_iter=50000):
    """Hill-climb using incremental validity. Try to add elements first, then swap."""
    cur = list(start)
    best = list(cur)
    cur_sums = _compute_sums(cur)
    pool = [x for x in range(1, 257) if x not in cur]
    stuck = 0
    
    while len(cur) < 30:
        if time.time() > deadline:
            break
        
        # Try to add a random element
        added = False
        for _ in range(30):
            if time.time() > deadline:
                break
            if not pool:
                break
            x = rng.choice(pool)
            new_sums, valid = _incremental_check_add(cur, x, cur_sums)
            if valid:
                cur.append(x)
                cur_sums = new_sums
                pool.remove(x)
                added = True
                if len(cur) > len(best):
                    best = list(cur)
                break
        
        if not added:
            # Try swap: remove one, add another
            if not pool:
                break
            swapped = False
            for _ in range(30):
                if time.time() > deadline:
                    break
                idx = rng.randrange(len(cur))
                x = rng.choice(pool)
                # Remove cur[idx], add x
                cur_without = cur[:idx] + cur[idx+1:]
                sums_without = _compute_sums(cur_without)
                new_sums, valid = _incremental_check_add(cur_without, x, sums_without)
                if valid:
                    cur = cur_without + [x]
                    cur_sums = new_sums
                    pool.remove(x)
                    pool.append(cur_without[idx])
                    swapped = True
                    if len(cur) > len(best):
                        best = list(cur)
                    break
            
            if not swapped:
                stuck += 1
                if stuck > 100:
                    break
        else:
            stuck = 0
    
    return best

def construct():
    deadline = time.time() + 85  # safety margin under 90s
    best = []
    
    # Pre-filter pool: remove numbers that are multiples of others in a basic sense
    # Actually, let's just use the full pool but with smart ordering
    
    for seed in range(200):
        if time.time() > deadline:
            break
        r = random.Random(seed)
        
        # Strategy 1: Greedy with random pool order
        pool = list(range(1, 257))
        r.shuffle(pool)
        cand = _greedy_build_incremental(r, pool, max_size=30, deadline=deadline)
        if len(cand) > len(best):
            best = list(cand)
        
        # Local search from greedy result
        if len(cand) >= 2:
            improved = _local_search_incremental(cand, r, deadline, max_iter=20000)
            if len(improved) > len(best):
                best = list(improved)
        
        # Strategy 2: Try with different pool orderings (biased toward larger numbers)
        if time.time() < deadline:
            pool2 = list(range(1, 257))
            # Sort by some heuristic: prefer numbers with fewer divisors
            def num_divisors(n):
                count = 0
                for i in range(1, int(n**0.5) + 1):
                    if n % i == 0:
                        count += 1
                        if i != n // i:
                            count += 1
                return count
            pool2.sort(key=lambda x: (num_divisors(x), -x))
            r.shuffle(pool2)
            cand2 = _greedy_build_incremental(r, pool2, max_size=30, deadline=deadline)
            if len(cand2) > len(best):
                best = list(cand2)
            if len(cand2) >= 2:
                improved2 = _local_search_incremental(cand2, r, deadline, max_iter=10000)
                if len(improved2) > len(best):
                    best = list(improved2)
        
        if len(best) >= 30:
            break
    
    # Final polish
    if len(best) < 30 and best:
        final = _local_search_incremental(best, random.Random(999), deadline, max_iter=50000)
        if len(final) > len(best):
            best = list(final)
    
    # Ensure Python ints and list
    result = [int(x) for x in best]
    assert len(set(result)) == len(result), "duplicates"
    assert all(1 <= x <= 256 for x in result), "out of range"
    return result

# EVOLVE-BLOCK-END
