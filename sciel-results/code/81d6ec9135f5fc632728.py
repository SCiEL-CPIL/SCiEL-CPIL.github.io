"""Rewritten program: greedy + local search for divisibility-incomparable subset sums."""
import time
import random
from functools import reduce
from itertools import combinations

def _subset_sums_ok(sums_set, new_elem):
    """Check if adding new_elem to a set whose subset sums are sums_set keeps them
    all distinct and divisibility-incomparable."""
    new_sums = set(sums_set)
    for s in sums_set:
        ns = s + new_elem
        if ns in new_sums:
            return False, None
        new_sums.add(ns)
    # Check divisibility: for each pair (a,b) in new_sums, a != b, neither divides the other
    # But we only need to check new elements against old ones and among themselves
    # Actually we need full check since new sums could divide each other
    # Efficient: for each new sum, check against all existing
    for s in sums_set:
        ns = s + new_elem
        # Check ns against all sums
        for t in sums_set:
            if t != ns and t != s:
                a, b = min(ns, t), max(ns, t)
                if b % a == 0:
                    return False, None
        # Also check ns against s itself? No, ns = s + x > s, so s divides ns only if ns % s == 0
        if ns % s == 0 and ns != s:
            return False, None
    # Check new sums among themselves
    new_only = [s + new_elem for s in sums_set]
    for i in range(len(new_only)):
        for j in range(i + 1, len(new_only)):
            a, b = min(new_only[i], new_only[j]), max(new_only[i], new_only[j])
            if b % a == 0:
                return False, None
    # Check new sums against old sums (already partially done above, but let's be thorough)
    for ns in new_only:
        for t in sums_set:
            if t != ns:
                a, b = min(ns, t), max(ns, t)
                if b % a == 0:
                    return False, None
    return True, new_sums

def _full_validity(elements):
    """Check full validity of a set of elements."""
    if not elements:
        return True
    # Compute all subset sums
    sums = set()
    for x in elements:
        new_sums = set()
        for s in sums:
            ns = s + x
            new_sums.add(ns)
        # Check for duplicates
        for ns in new_sums:
            if ns in sums:
                return False
        # Check divisibility among all pairs in sums | new_sums
        all_sums = sums | new_sums
        sums = all_sums
    # Final divisibility check
    slist = list(sums)
    for i in range(len(slist)):
        for j in range(i + 1, len(slist)):
            a, b = min(slist[i], slist[j]), max(slist[i], slist[j])
            if b % a == 0:
                return False
    return True

def _validity_check_fast(elements):
    """Faster validity check using incremental approach."""
    if not elements:
        return True
    sums = set()
    for x in elements:
        new_sums = set()
        for s in sums:
            ns = s + x
            new_sums.add(ns)
        # Check duplicates
        if new_sums & sums:
            return False
        # Check divisibility: for each ns in new_sums, check against all in sums
        for ns in new_sums:
            for t in sums:
                a, b = min(ns, t), max(ns, t)
                if b % a == 0:
                    return False
        # Check divisibility among new_sums
        ns_list = list(new_sums)
        for i in range(len(ns_list)):
            for j in range(i + 1, len(ns_list)):
                a, b = min(ns_list[i], ns_list[j]), max(ns_list[i], ns_list[j])
                if b % a == 0:
                    return False
        sums |= new_sums
    return True

def _greedy_build(elements):
    """Greedy: add elements one by one if valid."""
    result = []
    for x in elements:
        candidate = result + [x]
        if _validity_check_fast(candidate):
            result = candidate
    return result

def _local_search(elements, deadline, rng):
    """Local search: try to add more elements or swap to improve."""
    current = list(elements)
    current_set = set(current)
    
    # Build a pool of candidate elements not in current
    pool = [x for x in range(2, 257) if x not in current_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        rng.shuffle(pool)
        for x in pool:
            if time.time() >= deadline:
                break
            candidate = current + [x]
            if _validity_check_fast(candidate):
                current = candidate
                current_set.add(x)
                pool.remove(x)
                improved = True
    
    # Try swaps: remove one, add another
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i, x in enumerate(list(current)):
            if time.time() >= deadline:
                break
            for y in pool:
                candidate = [e for j, e in enumerate(current) if j != i] + [y]
                if len(candidate) > len(current) and _validity_check_fast(candidate):
                    current = candidate
                    current_set.discard(x)
                    current_set.add(y)
                    if y in pool:
                        pool.remove(y)
                    if x not in current_set:
                        pool.append(x)
                    improved = True
                    break
            if improved:
                break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Start with a known valid small set
    base = [31, 32, 34]
    
    # Try greedy from various orderings
    best = list(base)
    
    # Generate candidate orderings
    candidates = list(range(2, 257))
    
    # Try multiple random orderings
    rng = random.Random(42)
    
    # First pass: greedy with sorted order (larger first to avoid small divisors)
    for order in [sorted(candidates, reverse=True), sorted(candidates), list(candidates)]:
        if time.time() >= deadline:
            break
        result = _greedy_build(order)
        if len(result) > len(best):
            best = result
    
    # Local search improvement
    if time.time() < deadline - 5.0:
        improved = _local_search(best, deadline, rng)
        if len(improved) > len(best):
            best = improved
    
    # Final validation
    if not _validity_check_fast(best):
        # Fallback to base
        best = list(base)
    
    # Ensure return type is list of Python ints
    return [int(x) for x in best]
