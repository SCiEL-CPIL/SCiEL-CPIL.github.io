"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

# Precompute prime factorizations for all numbers in [2, 40]
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_NUM_PRIMES = len(_PRIMES)
_PRIME_IDX = {p: i for i, p in enumerate(_PRIMES)}

def _factorize(num):
    """Return list of (prime_index, exponent) for num."""
    factors = []
    x = num
    for i, p in enumerate(_PRIMES):
        if x % p == 0:
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            factors.append((i, cnt))
    return factors

# Precompute factorizations as list of (prime_idx, exp)
_FACTS = {n: _factorize(n) for n in range(2, 41)}

def _has_distinct_subset_sums_fast(lst):
    """Check if all subset sums of lst are distinct using bitmask enumeration."""
    k = len(lst)
    if k <= 1:
        return True
    seen = set()
    for mask in range(1 << k):
        s = 0
        for i in range(k):
            if mask & (1 << i):
                s += lst[i]
        if s in seen:
            return False
        seen.add(s)
    return True

def _can_add_fast(exps, cand_facts):
    """Check if adding cand_facts to exps keeps all subset sums distinct per prime."""
    for pi, e in cand_facts:
        if e == 0:
            continue
        new_list = exps[pi] + [e]
        if not _has_distinct_subset_sums_fast(new_list):
            return False
    return True

def _compute_exps(num_set):
    """Compute exponent lists per prime index for a set of numbers."""
    exps = [[] for _ in range(_NUM_PRIMES)]
    for num in num_set:
        for pi, e in _FACTS[num]:
            exps[pi].append(e)
    return exps

def _greedy_build(order, exps, current):
    """Greedy construction with given candidate ordering, mutating exps and current."""
    used = set(current)
    for cand in order:
        if len(current) >= 18:
            break
        if cand in used:
            continue
        facts = _FACTS[cand]
        if _can_add_fast(exps, facts):
            current.append(cand)
            used.add(cand)
            for pi, e in facts:
                exps[pi].append(e)
    return current

def _backtracking_search(start_set, remaining_candidates, target_size, deadline):
    """Backtracking to try to reach target_size with better pruning."""
    if len(start_set) >= target_size:
        return start_set[:]
    if time.time() > deadline:
        return start_set[:]
    if not remaining_candidates:
        return start_set[:]
    
    exps = _compute_exps(start_set)
    used = set(start_set)
    
    def _bt(current, exps_local, rem, depth):
        if len(current) >= target_size:
            return current[:]
        if time.time() > deadline:
            return None
        if not rem:
            return None
        
        # Sort candidates by "conflict potential" - prefer those with fewer primes
        # to leave more room for future additions
        rem_sorted = sorted(rem, key=lambda c: len(_FACTS[c]))
        
        for i, cand in enumerate(rem_sorted):
            if cand in used:
                continue
            facts = _FACTS[cand]
            if _can_add_fast(exps_local, facts):
                new_exps = [lst[:] for lst in exps_local]
                for pi, e in facts:
                    new_exps[pi].append(e)
                new_rem = [c for c in rem if c != cand]
                result = _bt(current + [cand], new_exps, new_rem, depth + 1)
                if result is not None:
                    return result
        return None
    
    return _bt(start_set[:], exps, remaining_candidates, 0)

def _try_randomized_greedy(deadline):
    """Try multiple randomized greedy orderings."""
    candidates = [n for n in range(4, 41) if n not in _PRIMES]
    best = []
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Create a randomized ordering with some structure
        # Mix: 50% sorted by size, 30% random, 20% sorted by factor count
        r = random.random()
        if r < 0.4:
            order = sorted(candidates, key=lambda x: (len(_FACTS[x]), x))
        elif r < 0.6:
            order = sorted(candidates, key=lambda x: (len(_FACTS[x]), -x))
        elif r < 0.8:
            order = sorted(candidates, reverse=True)
        else:
            order = candidates[:]
            random.shuffle(order)
        
        # Start with primes
        current = list(_PRIMES)
        exps = _compute_exps(current)
        _greedy_build(order, exps, current)
        
        if len(current) > len(best):
            best = current[:]
        if len(best) >= 18:
            break
    
    return best

def _try_backtracking_from_greedy(start_set, deadline):
    """Try to extend a greedy solution via backtracking."""
    if len(start_set) >= 18:
        return start_set[:]
    
    remaining = [n for n in range(2, 41) if n not in start_set]
    result = _backtracking_search(start_set, remaining, 18, deadline)
    return result if result is not None else start_set[:]

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    random.seed(42)
    
    # Phase 1: Randomized greedy search
    best = _try_randomized_greedy(deadline - 20)
    
    # Phase 2: Backtracking from best greedy result
    if len(best) < 18 and time.time() < deadline - 5:
        bt_result = _try_backtracking_from_greedy(best, deadline - 2)
        if bt_result is not None and len(bt_result) > len(best):
            best = bt_result
    
    # Phase 3: If still not 18, try more random restarts with backtracking
    if len(best) < 18 and time.time() < deadline - 10:
        candidates = [n for n in range(4, 41) if n not in _PRIMES]
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            order = candidates[:]
            random.shuffle(order)
            
            current = list(_PRIMES)
            exps = _compute_exps(current)
            _greedy_build(order, exps, current)
            
            if len(current) > len(best):
                best = current[:]
            
            # Try backtracking from this greedy result
            if len(current) < 18 and time.time() < deadline - 5:
                bt_res = _try_backtracking_from_greedy(current, deadline - 3)
                if bt_res is not None and len(bt_res) > len(best):
                    best = bt_res
            
            if len(best) >= 18:
                break
    
    # Final safety check
    if time.time() > deadline:
        pass  # Just return what we have
    
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
