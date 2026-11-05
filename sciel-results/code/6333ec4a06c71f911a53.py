"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, combinations_with_replacement

def _sidon_ok(S):
    """Check if S is a Sidon set (unique pair sums including a+a)."""
    if len(S) < 2:
        return True
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _diffs_of(S):
    """Get nonzero differences of S."""
    d = set()
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            d.add(S[j] - S[i])
            d.add(S[i] - S[j])
    return d

def _try_add(S, x, existing_sums, existing_diffs, other_diffs):
    """Try adding x to S, updating sums and diffs. Returns (ok, new_sums, new_diffs)."""
    new_sums = set(existing_sums)
    new_diffs = set(existing_diffs)
    
    # Check pair sums
    for a in S:
        s = a + x
        if s in new_sums:
            return False, None, None
        new_sums.add(s)
    # Check x+x
    s = x + x
    if s in new_sums:
        return False, None, None
    new_sums.add(s)
    
    # Check differences
    for a in S:
        d1 = x - a
        d2 = a - x
        if d1 in new_diffs or d2 in new_diffs:
            return False, None, None
        if d1 in other_diffs or d2 in other_diffs:
            return False, None, None
        new_diffs.add(d1)
        new_diffs.add(d2)
    
    return True, new_sums, new_diffs

def construct():
    deadline = time.time() + 55.0  # Safety margin before 90s limit
    best_sets = ([], [])
    best_size = 0
    
    def _greedy_one_ordering(order, max_time):
        """Greedy construction following a specific ordering of candidates."""
        sets = [[], []]
        sums = [set(), set()]
        diffs = [set(), set()]
        
        for x in order:
            if time.time() > max_time:
                break
            for side in (0, 1):
                ok, new_sums, new_diffs = _try_add(
                    sets[side], x, sums[side], diffs[side], diffs[1 - side]
                )
                if ok:
                    sets[side].append(x)
                    sums[side] = new_sums
                    diffs[side] = new_diffs
                    break  # Move to next x
        
        return sets
    
    # Strategy 1: Greedy from 1 to 256
    t0 = time.time()
    sets1 = _greedy_one_ordering(range(1, 257), deadline - 5)
    size1 = len(sets1[0]) + len(sets1[1])
    if size1 > best_size:
        best_sets = sets1
        best_size = size1
    
    # Strategy 2: Multiple random orderings
    random.seed(42)
    num_random = 20
    for trial in range(num_random):
        if time.time() > deadline - 5:
            break
        order = list(range(1, 257))
        random.shuffle(order)
        sets_r = _greedy_one_ordering(order, deadline - 5)
        size_r = len(sets_r[0]) + len(sets_r[1])
        if size_r > best_size:
            best_sets = sets_r
            best_size = size_r
    
    # Strategy 3: Greedy with bias toward smaller numbers first, then larger
    if time.time() < deadline - 5:
        order3 = list(range(1, 129)) + list(range(129, 257))
        random.shuffle(order3)
        sets3 = _greedy_one_ordering(order3, deadline - 5)
        size3 = len(sets3[0]) + len(sets3[1])
        if size3 > best_size:
            best_sets = sets3
            best_size = size3
    
    # Ensure nonempty
    if not best_sets[0] and not best_sets[1]:
        best_sets = [[1], [2]]
    
    return best_sets
# EVOLVE-BLOCK-END
