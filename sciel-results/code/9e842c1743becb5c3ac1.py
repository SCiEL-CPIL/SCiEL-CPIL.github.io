"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time
from itertools import combinations

_N = 128
_DEADLINE = 55.0  # safety margin under 60s kill

# Precompute LCM table for O(1) lookup
_LCM_TABLE = [[0] * (_N + 1) for _ in range(_N + 1)]
for i in range(1, _N + 1):
    for j in range(i, _N + 1):
        l = i // math.gcd(i, j) * j
        _LCM_TABLE[i][j] = l
        _LCM_TABLE[j][i] = l

# Precompute: for each pair (a,b), store the LCM value
# Also precompute which triples are bad
_BAD_TRIPLE_SET = set()
for a in range(1, _N + 1):
    for b in range(a + 1, _N + 1):
        ab = _LCM_TABLE[a][b]
        for c in range(b + 1, _N + 1):
            if _LCM_TABLE[a][c] == ab and _LCM_TABLE[b][c] == ab:
                _BAD_TRIPLE_SET.add((a, b, c))

# For each element x, precompute which pairs (a,b) with a<b would form a bad triple with x
# i.e., (a,b,x) is in _BAD_TRIPLE_SET in some order
def _build_conflict_pairs():
    """For each x, return set of (a,b) pairs such that {a,b,x} is a bad triple."""
    conflicts = [set() for _ in range(_N + 1)]
    for (a, b, c) in _BAD_TRIPLE_SET:
        # For element a: conflicts with pair (b,c)
        conflicts[a].add((b, c))
        conflicts[a].add((c, b))
        # For element b: conflicts with pair (a,c)
        conflicts[b].add((a, c))
        conflicts[b].add((c, a))
        # For element c: conflicts with pair (a,b)
        conflicts[c].add((a, b))
        conflicts[c].add((b, a))
    return conflicts

_CONFLICTS = _build_conflict_pairs()

def _bad_triple_exists(selected, x, lcm_table):
    """Check if adding x creates a bad triple with two existing elements."""
    s = selected
    n = len(s)
    # Optimization: Check pairs in s. If lcm(a,b) == lcm(a,x) == lcm(b,x), it's bad.
    # We can speed this up by checking lcm(a,x) first.
    for i in range(n):
        a = s[i]
        la_x = lcm_table[a][x]
        for j in range(i + 1, n):
            b = s[j]
            # If lcm(a,b) != la_x, it can't be equal to lcm(b,x) which must also be la_x
            if lcm_table[a][b] == la_x:
                if lcm_table[b][x] == la_x:
                    return True
    return False

def _greedy(order, lcm_table):
    selected = []
    for x in order:
        if not _bad_triple_exists(selected, x, lcm_table):
            selected.append(x)
    return selected

def _local_search_1_for_1(selected, base, lcm_table, rng, time_limit):
    """Try 1-for-1 swaps: remove one element, add another to increase or maintain size."""
    selected_set = set(selected)
    missing = [x for x in base if x not in selected_set]
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Shuffle to avoid getting stuck
        rng.shuffle(missing)
        rng.shuffle(selected)
        for i, remove_x in enumerate(selected):
            if time.time() > time_limit:
                break
            for add_y in missing:
                # Try swapping: remove remove_x, add add_y
                new_selected = selected[:i] + selected[i+1:] + [add_y]
                new_selected_set = set(new_selected)
                # Check if new_selected is valid
                valid = True
                for a, b, c in combinations(new_selected, 3):
                    if _LCM_TABLE[a][b] == _LCM_TABLE[a][c] == _LCM_TABLE[b][c]:
                        valid = False
                        break
                if valid and len(new_selected) > len(selected):
                    selected = new_selected
                    selected_set = new_selected_set
                    missing = [x for x in base if x not in selected_set]
                    improved = True
                    break
            if improved:
                break
    return selected

def _local_search_1_for_2(selected, base, lcm_table, rng, time_limit):
    """Try 1-for-2 swaps: remove one element, add two others to increase size by 1."""
    selected_set = set(selected)
    missing = [x for x in base if x not in selected_set]
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        rng.shuffle(missing)
        rng.shuffle(selected)
        for i, remove_x in enumerate(selected):
            if time.time() > time_limit:
                break
            # Try adding pairs from missing
            for j in range(len(missing)):
                if time.time() > time_limit:
                    break
                y1 = missing[j]
                for k in range(j + 1, len(missing)):
                    y2 = missing[k]
                    new_selected = selected[:i] + selected[i+1:] + [y1, y2]
                    # Check validity
                    valid = True
                    for a, b, c in combinations(new_selected, 3):
                        if _LCM_TABLE[a][b] == _LCM_TABLE[a][c] == _LCM_TABLE[b][c]:
                            valid = False
                            break
                    if valid and len(new_selected) > len(selected):
                        selected = new_selected
                        selected_set = set(new_selected)
                        missing = [x for x in base if x not in selected_set]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return selected

def _try_add_remaining(selected, base, lcm_table, time_limit):
    """Try to add any remaining elements that don't create bad triples."""
    selected_set = set(selected)
    for x in base:
        if time.time() > time_limit:
            break
        if x not in selected_set:
            if not _bad_triple_exists(selected, x, lcm_table):
                selected.append(x)
                selected_set.add(x)
    return selected

def construct():
    t0 = time.time()
    base = list(range(1, _N + 1))
    
    # Deterministic orders
    orders = [
        base,
        list(reversed(base)),
        sorted(base, key=lambda v: -v.bit_count()),
        sorted(base, key=lambda v: v.bit_count()),
        sorted(base, key=lambda v: -math.gcd(v, 128)),
        sorted(base, key=lambda v: v.bit_length()),
    ]
    
    best = []
    best_len = 0
    
    # Run greedy on deterministic orders
    for o in orders:
        if time.time() - t0 > _DEADLINE - 2:
            break
        cand = _greedy(o, _LCM_TABLE)
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)
            
    # Randomized Greedy with Restarts
    rng = random.Random(42)
    max_restarts = 3000
    for _ in range(max_restarts):
        if time.time() - t0 > _DEADLINE - 5:
            break
        
        # Random order
        o = base[:]
        rng.shuffle(o)
        
        cand = _greedy(o, _LCM_TABLE)
        
        # Apply local search to improve
        if len(cand) >= best_len:
            # 1-for-2 swap local search
            cand = _local_search_1_for_2(cand, base, _LCM_TABLE, rng, time.time() + 0.5)
            # Try to add remaining
            cand = _try_add_remaining(cand, base, _LCM_TABLE, time.time() + 0.1)
        
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)
            
            if best_len == _N:
                break

    # Final Polish: Try to add any remaining elements to the best set
    if time.time() - t0 < _DEADLINE - 1:
        best = _try_add_remaining(best, base, _LCM_TABLE, _DEADLINE - 0.5)

    if not best:
        best = [1]
        
    return best
# EVOLVE-BLOCK-END
