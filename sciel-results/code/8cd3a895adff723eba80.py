import time
import random
from math import isqrt

_START_TIME = time.time()
_DEADLINE = _START_TIME + 78.0

# Primes up to 40
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_NPRIMES = len(_PRIMES)

def _factor_vec(n):
    """Return the prime exponent vector of n as a tuple of length _NPRIMES."""
    vec = [0] * _NPRIMES
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            vec[i] += 1
            n //= p
    return tuple(vec)

# Precompute exponent vectors for all candidates 2..40
_EXP_VECTORS = {}
for _c in range(2, 41):
    _EXP_VECTORS[_c] = _factor_vec(_c)

def _subset_sums_set(vectors):
    """Return the set of all subset sums of vectors (including empty sum = zero vector)."""
    sums = {tuple([0] * _NPRIMES)}
    for v in vectors:
        new_sums = set()
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(_NPRIMES))
            new_sums.add(ns)
        sums = sums | new_sums
    return sums

def _can_add_vectors(sum_set, vec):
    """True iff adding vec keeps all subset sums distinct."""
    for s in sum_set:
        ns = tuple(s[i] + vec[i] for i in range(_NPRIMES))
        if ns in sum_set:
            return False
    return True

def _try_add_vectors(selected, sum_set, x):
    """Try to add x to selected; returns (success, new_selected, new_sum_set)."""
    vec = _EXP_VECTORS[x]
    if not _can_add_vectors(sum_set, vec):
        return False, selected, sum_set
    new_sum_set = set(sum_set)
    for s in sum_set:
        ns = tuple(s[i] + vec[i] for i in range(_NPRIMES))
        new_sum_set.add(ns)
    new_selected = selected + [x]
    return True, new_selected, new_sum_set

def _construct_initial_set():
    """
    Construct a high-quality initial set using the 'pure powers' heuristic.
    Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    Squares: 4 (2^2), 9 (3^2), 25 (5^2)
    Fourth power: 16 (2^4)
    This set of 16 is known to be valid.
    """
    return [2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 19, 23, 25, 29, 31, 37]

def _subset_sums_from_selected(selected):
    """Compute the subset sums set from a list of selected integers."""
    vectors = [_EXP_VECTORS[x] for x in selected]
    return _subset_sums_set(vectors)

def _local_search(selected, sum_set, deadline, max_rounds=30):
    """Local search: try swapping elements or adding elements to improve the set size."""
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        if len(selected) >= 18:
            break
        improved = False
        used = set(selected)
        candidates = [c for c in range(2, 41) if c not in used]
        
        # 1. Try adding any candidate
        for x in candidates:
            if time.time() > deadline:
                break
            ok, new_sel, new_sum_set = _try_add_vectors(selected, sum_set, x)
            if ok:
                selected = new_sel
                sum_set = new_sum_set
                improved = True
                used = set(selected)
                candidates = [c for c in range(2, 41) if c not in used]
                if len(selected) >= 18:
                    break
        
        if not improved:
            # 2. Try single swaps: remove one, add one different element
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                removed = selected[i]
                remaining = selected[:i] + selected[i+1:]
                rem_sum_set = _subset_sums_from_selected(remaining)
                
                for x in candidates:
                    if time.time() > deadline:
                        break
                    ok1, sel1, sum_set1 = _try_add_vectors(remaining, rem_sum_set, x)
                    if ok1:
                        # Check if this is actually an improvement (different element)
                        if x != removed:
                            selected = sel1
                            sum_set = sum_set1
                            improved = True
                            break
                if improved:
                    break
            
        if not improved:
            # 3. Try swapping: remove one, add two
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                # Remove selected[i]
                remaining = selected[:i] + selected[i+1:]
                rem_sum_set = _subset_sums_from_selected(remaining)
                
                # Try adding two candidates
                added_any = False
                for j in range(len(candidates)):
                    if time.time() > deadline:
                        break
                    x1 = candidates[j]
                    ok1, sel1, sum_set1 = _try_add_vectors(remaining, rem_sum_set, x1)
                    if not ok1:
                        continue
                    for k in range(j+1, len(candidates)):
                        if time.time() > deadline:
                            break
                        x2 = candidates[k]
                        ok2, sel2, sum_set2 = _try_add_vectors(sel1, sum_set1, x2)
                        if ok2 and len(sel2) > len(selected):
                            selected = sel2
                            sum_set = sum_set2
                            improved = True
                            added_any = True
                            break
                    if added_any:
                        break
                if improved:
                    break
        
        if not improved:
            break
            
    return selected, sum_set

def construct():
    if time.time() > _DEADLINE:
        return [2]

    # Start with the strong heuristic set
    selected = _construct_initial_set()
    sum_set = _subset_sums_from_selected(selected)
    
    # Refine with local search
    if len(selected) < 18 and time.time() < _DEADLINE:
        selected, sum_set = _local_search(selected, sum_set, _DEADLINE)

    # Final greedy fill
    if time.time() < _DEADLINE and len(selected) < 18:
        used = set(selected)
        for x in range(2, 41):
            if x in used:
                continue
            if len(selected) >= 18:
                break
            ok, selected, sum_set = _try_add_vectors(selected, sum_set, x)
            if ok:
                used.add(x)

    # Ensure validity and type correctness
    # Re-verify the final set using exponent vectors
    final_sum_set = _subset_sums_from_selected(selected)
    if len(final_sum_set) != 2 ** len(selected):
        # Fallback to a known valid set if something went wrong
        selected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    # Deduplicate and filter
    seen = set()
    out = []
    for x in selected:
        if 2 <= x <= 40 and x not in seen:
            seen.add(x)
            out.append(int(x))
            
    if not out:
        out = [2]
        
    return out
