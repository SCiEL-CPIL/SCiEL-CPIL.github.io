"""Find the maximum-size dissociated subset of [1..22].

A subset S is dissociated (reciprocal) iff every subset of S has a distinct
sum of reciprocals: the map A -> sum_{a in A} 1/a is injective over all 2^|S|
sub-sums (including the empty sum = 0).

Strategy:
  - Represent reciprocal sums as integers scaled by LCM(1..22) for exact
    arithmetic without Fraction overhead.
  - Start from a known valid 9-element seed.
  - Use randomized local search with swap moves to escape local optima.
    A swap removes one element and adds another, potentially increasing
    cardinality or improving the set's "quality" for further growth.
  - Also attempt greedy construction from random orderings.
  - Global deadline guard with safety margin.
"""
import time
import random
from math import gcd
from functools import reduce

_T0 = time.time()
_BUDGET = 85.0  # wall-clock guard (evaluator kills at 90s; keep 5s margin)
_N = 22


def _lcm(a, b):
    return a // gcd(a, b) * b


_LCM = reduce(_lcm, range(1, _N + 1))
# Reciprocal of i as integer: LCM // i
_REC = [0] + [_LCM // i for i in range(1, _N + 1)]


def _is_dissociated(elements):
    """Independent verification: all 2^k reciprocal sub-sums of `elements` distinct."""
    sums = {0}
    for e in elements:
        v = _REC[e]
        shifted = {s + v for s in sums}
        if shifted & sums:
            return False
        sums |= shifted
    return True


def _can_add(sums_set, v):
    """True iff adding reciprocal v keeps every sum distinct from every other sum+v."""
    for s in sums_set:
        if (v - s) in sums_set:
            return False
    return True


def _compute_sums(elements):
    """Compute the set of all subset sums for a dissociated set."""
    sums = {0}
    for e in elements:
        v = _REC[e]
        shifted = {s + v for s in sums}
        sums |= shifted
    return sums


def _greedy_from_order(order, best_size):
    """Greedy construction: add elements in given order if they preserve dissociation."""
    current_set = []
    sums = {0}
    for e in order:
        v = _REC[e]
        if _can_add(sums, v):
            current_set.append(e)
            sums |= {s + v for s in sums}
            if len(current_set) > best_size:
                return current_set
    return current_set if len(current_set) > best_size else []


def _local_search(initial_set, deadline):
    """Randomized local search with swap moves to improve cardinality."""
    current_set = list(initial_set)
    current_sums = _compute_sums(current_set)
    best_set = list(current_set)
    best_size = len(current_set)
    
    used = set(current_set)
    available = [e for e in range(1, _N + 1) if e not in used]
    
    # First try to add any available element
    for e in available[:]:
        v = _REC[e]
        if _can_add(current_sums, v):
            current_set.append(e)
            current_sums = _compute_sums(current_set)
            used.add(e)
            available.remove(e)
            best_set = list(current_set)
            best_size = len(current_set)
    
    # Now try swaps: remove one, add another
    while time.time() < deadline:
        # Try random swaps
        if not available or not current_set:
            break
        
        # Try to add a random available element
        added = False
        for _ in range(min(50, len(available))):
            e = random.choice(available)
            v = _REC[e]
            if _can_add(current_sums, v):
                current_set.append(e)
                current_sums = _compute_sums(current_set)
                used.add(e)
                available.remove(e)
                best_set = list(current_set)
                best_size = len(current_set)
                added = True
                break
        
        if added:
            continue
        
        # Try swaps: remove one from current, add one from available
        improved = False
        for _ in range(min(100, len(current_set) * len(available))):
            rem_idx = random.randrange(len(current_set))
            rem = current_set[rem_idx]
            add = random.choice(available)
            
            if rem == add:
                continue
            
            # Try the swap
            new_set = current_set[:rem_idx] + current_set[rem_idx+1:] + [add]
            if _is_dissociated(new_set):
                # Check if this improves things (same size, but maybe better for future adds)
                # For now, accept it and see if we can add more
                new_sums = _compute_sums(new_set)
                new_used = set(new_set)
                new_available = [e for e in range(1, _N + 1) if e not in new_used]
                
                # Try to add more elements to the new set
                can_add_more = False
                for e in new_available[:20]:
                    v = _REC[e]
                    if _can_add(new_sums, v):
                        can_add_more = True
                        break
                
                # Accept the swap if it allows adding more or is same size
                if can_add_more or len(new_set) > len(current_set):
                    current_set = new_set
                    current_sums = new_sums
                    used = new_used
                    available = new_available
                    improved = True
                    break
        
        if not improved:
            # Try removing a random element to free up space for potentially better additions
            if len(current_set) > best_size - 2:
                rem_idx = random.randrange(len(current_set))
                rem = current_set[rem_idx]
                current_set = current_set[:rem_idx] + current_set[rem_idx+1:]
                current_sums = _compute_sums(current_set)
                used = set(current_set)
                available = [e for e in range(1, _N + 1) if e not in used]
    
    return best_set


def _verified_baseline():
    # The known valid seed; keep it only if it really is dissociated (it is).
    seed = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    return seed if _is_dissociated(seed) else []


def construct():
    """Return a nonempty subset of [1..22] maximising dissociated cardinality."""
    fallback = _verified_baseline()
    best = list(fallback)
    best_size = len(best)
    
    deadline = _T0 + _BUDGET
    
    # Phase 1: Try greedy from multiple orderings
    orderings = [
        list(range(_N, 0, -1)),  # descending
        list(range(1, _N + 1)),  # ascending
        [i for i in range(_N, 0, -2)] + [i for i in range(_N - 1, 0, -2)],  # odd then even, descending
        [19, 17, 13, 11, 7, 5, 3, 2, 22, 21, 20, 18, 16, 15, 14, 12, 10, 9, 8, 6, 4, 1],
        [1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 2, 3, 5, 7, 11, 13, 17, 19],
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        result = _greedy_from_order(order, best_size)
        if result and len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Phase 2: Randomized greedy from random orderings
    while time.time() < deadline - 10:
        order = list(range(1, _N + 1))
        random.shuffle(order)
        result = _greedy_from_order(order, best_size)
        if result and len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Phase 3: Local search from the best set found
    if time.time() < deadline - 5:
        local_result = _local_search(best, deadline)
        if local_result and len(local_result) > best_size:
            best = local_result
            best_size = len(best)
    
    # Verify final result
    candidates = [c for c in (best, fallback) if c and _is_dissociated(c)]
    if not candidates:
        candidates = [[1]]
    candidates.sort(key=len, reverse=True)
    return candidates[0]
