import time
import random
from itertools import combinations

def _subset_sums_set(elements):
    """Return the set of all non-empty subset sums."""
    n = len(elements)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _is_divisibility_free(sums):
    """Check no two distinct sums divide one another."""
    if len(sums) < 2:
        return True
    sorted_sums = sorted(sums)
    for i, a in enumerate(sorted_sums):
        for j in range(i + 1, len(sorted_sums)):
            if sorted_sums[j] % a == 0:
                return False
    return True

def _is_valid(elements):
    """Full validation."""
    if not elements:
        return True
    if len(elements) != len(set(elements)):
        return False
    for x in elements:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if not (1 <= x <= 256):
            return False
    sums = _subset_sums_set(elements)
    return _is_divisibility_free(sums)

def _build_blocked(sums, max_val):
    """Build a set of all values in 1..max_val that are multiples of any sum in sums."""
    blocked = set()
    for s in sums:
        for m in range(2 * s, max_val + 1, s):
            blocked.add(m)
    return blocked

def _check_add_fast(elements, candidate, sums, blocked, max_val):
    """
    Fast check if adding candidate to elements keeps the set valid.
    Uses precomputed blocked set.
    Returns (new_sums, new_blocked, True) if valid, (None, None, False) otherwise.
    """
    if candidate in elements:
        return None, None, False
    
    # Check: candidate must not be a multiple of any existing sum
    if candidate in blocked:
        return None, None, False
    
    # Compute new sums
    new_sums = set(sums)
    for s in sums:
        new_sums.add(s + candidate)
    new_sums.add(candidate)
    
    # Check: no new sum divides another new sum, and no existing sum divides a new sum
    # Existing sums dividing new sums: new_sum = s + candidate or candidate
    # candidate already checked. For s+candidate: need (s+candidate) % t != 0 for all t in sums
    # New sums dividing each other: for t1 = s1+c, t2 = s2+c, need t1 !| t2
    
    # First check: no existing sum divides any new sum
    for s in sums:
        t = s + candidate
        # Check if any existing sum divides t
        for u in sums:
            if u != s and t % u == 0:
                return None, None, False
        # Check if candidate divides t (i.e., t % candidate == 0)
        if t % candidate == 0:
            return None, None, False
    
    # Check: no new sum divides another new sum
    new_only = set()
    for s in sums:
        new_only.add(s + candidate)
    new_only.add(candidate)
    
    sorted_new = sorted(new_only)
    for i, a in enumerate(sorted_new):
        for j in range(i + 1, len(sorted_new)):
            if sorted_new[j] % a == 0:
                return None, None, False
    
    # Build new blocked set
    new_blocked = set(blocked)
    for s in new_sums:
        for m in range(2 * s, max_val + 1, s):
            new_blocked.add(m)
    
    return new_sums, new_blocked, True

def _greedy_grow(elements, candidates, sums, deadline):
    """Greedily add candidates in given order, using fast checks."""
    elements = list(elements)
    sums = set(sums)
    max_val = 2 * 256 * 30  # max possible sum is 30*256=7680, but we need up to 2*7680
    max_val = 15360
    blocked = _build_blocked(sums, max_val)
    
    for c in candidates:
        if time.time() > deadline:
            break
        if c in elements:
            continue
        new_sums, new_blocked, ok = _check_add_fast(elements, c, sums, blocked, max_val)
        if ok:
            elements.append(c)
            sums = new_sums
            blocked = new_blocked
    return elements, sums

def _backtrack(elements, sums, target_size, deadline, best_ref, candidate_pool):
    """
    Backtracking search to try to reach target_size.
    best_ref is a list [best_elements] that gets updated.
    """
    if len(elements) > len(best_ref[0]):
        best_ref[0] = list(elements)
    if len(elements) >= target_size:
        return
    if time.time() > deadline:
        return
    
    max_val = 15360
    blocked = _build_blocked(sums, max_val)
    
    for i, c in enumerate(candidate_pool):
        if time.time() > deadline:
            return
        if c in elements:
            continue
        new_sums, new_blocked, ok = _check_add_fast(elements, c, sums, blocked, max_val)
        if ok:
            _backtrack(elements + [c], new_sums, target_size, deadline, best_ref, candidate_pool[i+1:])
            if time.time() > deadline:
                return

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    random.seed(42)
    all_nums = list(range(1, 257))
    
    best_result = [31, 32, 34]
    
    # Strategy 1: Start with small numbers (fewer multiples to block) and greedily grow
    for base in range(1, 50):
        if time.time() > deadline:
            break
        start = [base]
        sums = _subset_sums_set(start)
        # Try adding other small numbers first
        nearby = sorted(all_nums, key=lambda x: abs(x - base))
        nearby = [x for x in nearby if x != base]
        result, _ = _greedy_grow(start, nearby, sums, deadline)
        if len(result) > len(best_result):
            best_result = list(result)
    
    # Strategy 2: Start with pairs of small numbers
    for a in range(1, 30, 2):
        if time.time() > deadline:
            break
        for b in range(a + 1, min(a + 20, 30)):
            if time.time() > deadline:
                break
            start = [a, b]
            sums = _subset_sums_set(start)
            nearby = sorted(all_nums, key=lambda x: min(abs(x - a), abs(x - b)))
            nearby = [x for x in nearby if x not in start]
            result, _ = _greedy_grow(start, nearby, sums, deadline)
            if len(result) > len(best_result):
                best_result = list(result)
    
    # Strategy 3: Randomized restarts with small-first ordering
    for trial in range(2000):
        if time.time() > deadline:
            break
        
        k = random.randint(1, 4)
        start = random.sample(all_nums, k)
        sums = _subset_sums_set(start)
        
        remaining = [x for x in all_nums if x not in start]
        # Sort by value (smaller first) with some randomness
        random.shuffle(remaining)
        remaining.sort(key=lambda x: (x // 10, random.random()))
        
        result, _ = _greedy_grow(start, remaining, sums, deadline)
        if len(result) > len(best_result):
            best_result = list(result)
    
    # Strategy 4: Local search / hill climbing on the best found
    if len(best_result) >= 3 and time.time() < deadline - 10:
        current = list(best_result)
        sums = _subset_sums_set(current)
        max_val = 15360
        blocked = _build_blocked(sums, max_val)
        
        for iteration in range(3000):
            if time.time() > deadline:
                break
            
            improved = False
            used = set(current)
            candidates = [x for x in all_nums if x not in used]
            random.shuffle(candidates)
            
            for c in candidates:
                new_sums, new_blocked, ok = _check_add_fast(current, c, sums, blocked, max_val)
                if ok:
                    current.append(c)
                    sums = new_sums
                    blocked = new_blocked
                    improved = True
                    break
            
            if not improved:
                if len(current) >= 2:
                    remove_idx = random.randint(0, len(current) - 1)
                    removed = current.pop(remove_idx)
                    new_sums = _subset_sums_set(current)
                    new_blocked = _build_blocked(new_sums, max_val)
                    
                    used = set(current)
                    candidates = [x for x in all_nums if x not in used]
                    random.shuffle(candidates)
                    
                    added_any = False
                    for c in candidates:
                        ns, nb, ok = _check_add_fast(current, c, new_sums, new_blocked, max_val)
                        if ok:
                            current.append(c)
                            new_sums = ns
                            new_blocked = nb
                            added_any = True
                            break
                    
                    if not added_any:
                        current.append(removed)
                        new_sums = _subset_sums_set(current)
                        new_blocked = _build_blocked(new_sums, max_val)
                    else:
                        sums = new_sums
                        blocked = new_blocked
                        if len(current) > len(best_result):
                            best_result = list(current)
    
    # Strategy 5: Targeted backtracking from promising starts
    if time.time() < deadline - 5:
        if len(best_result) < 10:
            target = len(best_result) + 1
            best_ref = [list(best_result)]
            used = set(best_result)
            pool = [x for x in all_nums if x not in used]
            pool.sort(key=lambda x: x)  # small first
            sums = _subset_sums_set(best_result)
            _backtrack(list(best_result), sums, target, deadline, best_ref, pool)
            if len(best_ref[0]) > len(best_result):
                best_result = list(best_ref[0])
    
    # Final validation
    if _is_valid(best_result):
        return [int(x) for x in best_result]
    
    return [31, 32, 34]
