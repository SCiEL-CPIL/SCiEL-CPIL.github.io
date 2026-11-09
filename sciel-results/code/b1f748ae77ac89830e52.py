"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _subset_sums_bitset(elements):
    """Return a set of all non-empty subset sums."""
    sums = set()
    n = len(elements)
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _check_divisibility(sums):
    """Check no two distinct sums divide each other."""
    s = sorted(sums)
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if s[j] % s[i] == 0:
                return False
    return True

def _is_valid(elements):
    """Full validity check."""
    if len(elements) != len(set(elements)):
        return False
    if any(x < 1 or x > 256 for x in elements):
        return False
    sums = _subset_sums_bitset(elements)
    if len(sums) != sum(1 << i for i in range(len(elements))):
        # This check is wrong; let's just check uniqueness properly
        pass
    # Check uniqueness: if any duplicate sums exist, invalid
    # We already use a set, so duplicates are collapsed. But the rule says
    # two different subsets with same sum makes it invalid.
    # So we need to detect if two different subsets produce the same sum.
    # Let's count subset sums with multiplicity.
    n = len(elements)
    sum_count = {}
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sum_count[s] = sum_count.get(s, 0) + 1
    if any(v > 1 for v in sum_count.values()):
        return False
    return _check_divisibility(list(sum_count.keys()))

def _try_extend(current, candidate, deadline):
    """Try adding candidate to current list, return new list or None."""
    if time.time() > deadline:
        return None
    new_list = current + [candidate]
    if _is_valid(new_list):
        return new_list
    return None

def _greedy_build(candidates, deadline, max_size=30):
    """Greedy: add candidates in given order if valid."""
    current = []
    for c in candidates:
        if time.time() > deadline:
            break
        if len(current) >= max_size:
            break
        new_list = _try_extend(current, c, deadline)
        if new_list is not None:
            current = new_list
    return current

def _local_search_improve(best_set, deadline):
    """Try to improve by swapping or adding elements."""
    current = list(best_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding a new element
        for c in range(1, 257):
            if time.time() > deadline:
                break
            if c in current:
                continue
            new_list = current + [c]
            if len(new_list) > len(current) and _is_valid(new_list):
                current = new_list
                improved = True
                break
        # Try swapping: remove one, add another
        if not improved:
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for c in range(1, 257):
                    if time.time() > deadline:
                        break
                    if c in current:
                        continue
                    new_list = current[:i] + [c] + current[i+1:]
                    if _is_valid(new_list):
                        # Check if this swap enables adding more later
                        # For now, accept any valid swap that doesn't reduce size
                        # Actually, we want to find a swap that allows further growth
                        # Let's try: after swap, can we add something?
                        for c2 in range(1, 257):
                            if time.time() > deadline:
                                break
                            if c2 in new_list:
                                continue
                            if _is_valid(new_list + [c2]):
                                current = new_list + [c2]
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    best_set = [31, 32, 34]
    
    # Strategy 1: Greedy with large numbers first (descending)
    cands_desc = list(range(256, 0, -1))
    result = _greedy_build(cands_desc, deadline)
    if len(result) > len(best_set):
        best_set = result
    
    # Strategy 2: Multiple randomized greedy attempts
    for attempt in range(500):
        if time.time() > deadline:
            break
        cands = list(range(1, 257))
        random.shuffle(cands)
        result = _greedy_build(cands, deadline)
        if len(result) > len(best_set):
            best_set = result
            if len(best_set) >= 30:
                break
    
    # Strategy 3: Local search improvement
    if time.time() < deadline:
        improved_set = _local_search_improve(best_set, deadline)
        if len(improved_set) > len(best_set):
            best_set = improved_set
    
    # Strategy 4: Try specific known-good starting points and extend
    # Start with various small valid sets and try to grow them
    if time.time() < deadline:
        starting_sets = [
            [128, 192, 224],
            [100, 150, 200],
            [64, 96, 128, 160],
            [80, 120, 160, 200, 240],
            [50, 80, 110, 140, 170, 200],
        ]
        for ss in starting_sets:
            if time.time() > deadline:
                break
            if not _is_valid(ss):
                continue
            # Try to extend from this starting point
            cands = [c for c in range(1, 257) if c not in ss]
            random.shuffle(cands)
            current = list(ss)
            for c in cands:
                if time.time() > deadline:
                    break
                if len(current) >= 30:
                    break
                new_list = _try_extend(current, c, deadline)
                if new_list is not None:
                    current = new_list
            if len(current) > len(best_set):
                best_set = current
                if len(best_set) >= 30:
                    break
    
    # Final local search
    if time.time() < deadline - 5:
        improved_set = _local_search_improve(best_set, deadline)
        if len(improved_set) > len(best_set):
            best_set = improved_set
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
