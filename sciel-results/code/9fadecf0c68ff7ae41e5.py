"""Program for large reciprocal-subsum-free sets using branch-and-bound + greedy."""
import time
import math
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

# Precompute LCM of 1..24
_LCM_24 = 1
for i in range(1, 25):
    _LCM_24 = _LCM_24 * i // math.gcd(_LCM_24, i)

# Precompute scaled reciprocals
_V = [_LCM_24 // i for i in range(1, 25)] # V[i-1] = LCM/i


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _lcm(a, b):
    return a * b // math.gcd(a, b)


def _can_form_sum(target, items):
    """
    Check if target can be formed as a sum of a subset of items.
    Uses meet-in-the-middle with early termination.
    """
    n = len(items)
    if n == 0:
        return target == 0
    if target == 0:
        return True
    total = sum(items)
    if total < target:
        return False
    if total == target:
        return True
    
    mid = n // 2
    left = items[:mid]
    right = items[mid:]
    
    left_sums = {0}
    for x in left:
        new_sums = set()
        for s in left_sums:
            ns = s + x
            if ns <= target:
                new_sums.add(ns)
        left_sums.update(new_sums)
    
    right_sums = {0}
    for x in right:
        new_sums = set()
        for s in right_sums:
            ns = s + x
            if ns <= target:
                new_sums.add(ns)
        right_sums.update(new_sums)
    
    # Check if any left sum + right sum = target
    for s in left_sums:
        if (target - s) in right_sums:
            return True
            
    return False


def _is_subsum_free(S):
    """
    Check if set S is reciprocal-subsum-free.
    Uses precomputed _V.
    """
    if len(S) < 2:
        return True
    
    S_list = list(S)
    n = len(S_list)
    
    vals = [_V[x-1] for x in S_list]
    
    for i in range(n):
        target = vals[i]
        others = [v for j, v in enumerate(vals) if j != i]
        
        if _can_form_sum(target, others):
            return False
            
    return True


def _can_add_element(current_vals, idx, val):
    """
    Check if adding element idx (with value val) to a set with current_vals
    maintains subsum-freeness. Only checks constraints involving the new element
    and verifies existing elements aren't violated by the new one.
    """
    # Check: can val be formed from current_vals?
    if _can_form_sum(val, current_vals):
        return False
    
    # Check: for each existing element, can its value now be formed
    # using the new element plus some of the others?
    # This is equivalent to: for each existing i, can current_vals[i] be formed
    # from (current_vals minus current_vals[i]) plus val?
    # i.e., can current_vals[i] - val be formed from (current_vals minus current_vals[i])?
    # Or can current_vals[i] be formed from (current_vals minus current_vals[i]) ∪ {val}?
    for i, cv in enumerate(current_vals):
        # Can cv be formed from (current_vals without cv) ∪ {val}?
        # This means: either cv can be formed without val (already false since set was valid),
        # OR cv = val + (subset of current_vals without cv)
        # So check: can (cv - val) be formed from (current_vals without cv)?
        if cv > val:
            remaining = [v for j, v in enumerate(current_vals) if j != i]
            if _can_form_sum(cv - val, remaining):
                return False
        elif cv == val:
            # cv = val, so we need empty subset of others - but that's trivially true
            # Actually if cv == val, then cv = val + 0, and 0 is always formable
            # But wait, we need a NON-EMPTY subset. The subset of others can be empty?
            # Re-reading: "sum of reciprocals of a subset of the OTHER selected elements"
            # Subset can be empty? No, typically "subset" in this context means non-empty.
            # Actually, if the subset is empty, sum = 0, which won't equal 1/k > 0.
            # So empty subset is fine. But cv == val means 1/k = 1/k, which would be
            # formed by just {val} itself, but val is the NEW element and cv is existing.
            # The constraint is: no reciprocal of a SELECTED element equals sum of reciprocals
            # of a subset of the OTHER selected elements.
            # So for existing element with value cv, others include the new element.
            # Can cv be formed from a subset of (current_vals without cv) ∪ {val}?
            # If cv == val, then {val} alone gives cv. So yes, this violates.
            return False
    return True


def _try_add_elements(current, candidates, max_add=3):
    """
    Try to add up to max_add elements from candidates to current while maintaining subsum-freeness.
    Returns the best extended set found.
    """
    if not candidates or not current:
        return current
    
    best = set(current)
    
    # Try adding 1 element
    for c in candidates:
        if _time_remaining() < 3.0:
            break
        new_set = best | {c}
        if _is_subsum_free(new_set):
            if len(new_set) > len(best):
                best = new_set
    
    # Try adding 2 elements
    if len(candidates) >= 2 and _time_remaining() > 5.0:
        for c1, c2 in combinations(candidates, 2):
            if _time_remaining() < 3.0:
                break
            new_set = best | {c1, c2}
            if _is_subsum_free(new_set):
                if len(new_set) > len(best):
                    best = new_set
    
    # Try adding 3 elements
    if len(candidates) >= 3 and _time_remaining() > 8.0:
        for c1, c2, c3 in combinations(candidates, 3):
            if _time_remaining() < 3.0:
                break
            new_set = best | {c1, c2, c3}
            if _is_subsum_free(new_set):
                if len(new_set) > len(best):
                    best = new_set
    
    return best


def _branch_and_bound(current_set, candidates, best_size, best_set, depth=0, max_depth=4):
    """
    Branch and bound: try to add elements from candidates to current_set.
    Returns (best_size, best_set) found.
    """
    if _time_remaining() < 3.0 or depth >= max_depth:
        return best_size, best_set
    
    if len(candidates) == 0:
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
        return best_size, best_set
    
    # Try adding each candidate
    for i, c in enumerate(candidates):
        if _time_remaining() < 3.0:
            break
        
        new_set = current_set | {c}
        if _is_subsum_free(new_set):
            remaining = [x for x in candidates if x != c]
            bs, bs_set = _branch_and_bound(new_set, remaining, best_size, best_set, depth + 1, max_depth)
            if bs > best_size:
                best_size = bs
                best_set = bs_set
    
    # Also consider not adding any more (current set is valid)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = set(current_set)
    
    return best_size, best_set


def _swap_improve(current):
    """
    Try to improve current set by swapping out one element for one or more others.
    Returns the best set found.
    """
    if not current or _time_remaining() < 5.0:
        return current
    
    best = set(current)
    all_elements = set(range(1, 25))
    missing = all_elements - best
    
    if not missing:
        return best
    
    # Try swapping out one element for one or more missing elements
    for elem_to_remove in list(best):
        if _time_remaining() < 3.0:
            break
        
        base_without = best - {elem_to_remove}
        
        # Try adding one missing element
        for m in missing:
            if _time_remaining() < 2.0:
                break
            new_set = base_without | {m}
            if _is_subsum_free(new_set):
                if len(new_set) > len(best):
                    best = new_set
        
        # Try adding two missing elements
        if len(missing) >= 2 and _time_remaining() > 5.0:
            for m1, m2 in combinations(missing, 2):
                if _time_remaining() < 2.0:
                    break
                new_set = base_without | {m1, m2}
                if _is_subsum_free(new_set):
                    if len(new_set) > len(best):
                        best = new_set
    
    return best


def _bbs_from_start(start_set, max_depth=3):
    """
    Run branch and bound starting from start_set, trying to add missing elements.
    """
    if not start_set or _time_remaining() < 10.0:
        return start_set
    
    all_elements = set(range(1, 25))
    missing = sorted(all_elements - start_set)
    
    if not missing:
        return start_set
    
    best_size = len(start_set)
    best_set = set(start_set)
    
    # Use B&B to try to extend
    _, best_set = _branch_and_bound(set(start_set), missing, best_size, best_set, 0, max_depth)
    
    return best_set


def _greedy_descending():
    """Greedy construction from 24 down to 1."""
    current = set()
    for k in range(24, 0, -1):
        if _time_remaining() < 5.0:
            break
        new_set = current | {k}
        if _is_subsum_free(new_set):
            current = new_set
    return current


def _greedy_ascending():
    """Greedy construction from 1 up to 24."""
    current = set()
    for k in range(1, 25):
        if _time_remaining() < 5.0:
            break
        new_set = current | {k}
        if _is_subsum_free(new_set):
            current = new_set
    return current


def _greedy_random(seed_val):
    """Greedy construction with random ordering."""
    import random
    rng = random.Random(seed_val)
    elems = list(range(1, 25))
    rng.shuffle(elems)
    current = set()
    for k in elems:
        if _time_remaining() < 5.0:
            break
        new_set = current | {k}
        if _is_subsum_free(new_set):
            current = new_set
    return current


def _local_search(current):
    """Apply local search improvements."""
    if not current:
        return current
    
    all_elements = set(range(1, 25))
    missing = all_elements - current
    
    if missing and _time_remaining() > 10.0:
        current = _try_add_elements(current, list(missing), max_add=3)
    
    if _time_remaining() > 15.0:
        current = _swap_improve(current)
    
    missing = all_elements - current
    if missing and _time_remaining() > 8.0:
        current = _try_add_elements(current, list(missing), max_add=2)
    
    return current


def _run_bbs_search(seed_val, max_depth=3):
    """
    Run a B&B search starting from a random greedy solution.
    """
    if _time_remaining() < 15.0:
        return set()
    
    rng = random.Random(seed_val)
    elems = list(range(1, 25))
    rng.shuffle(elems)
    
    # Build a greedy starting set
    current = set()
    for k in elems:
        if _time_remaining() < 5.0:
            break
        new_set = current | {k}
        if _is_subsum_free(new_set):
            current = new_set
    
    if not current:
        return set()
    
    # Now try to extend with B&B
    return _bbs_from_start(current, max_depth)


def construct():
    """
    Return a nonempty subset of [1,24] that is reciprocal-subsum-free,
    maximizing cardinality.
    """
    best = set()
    
    # Try multiple greedy strategies and keep the best
    strategies = [
        _greedy_descending,
        _greedy_ascending,
    ]
    
    for strategy in strategies:
        if _time_remaining() < 10.0:
            break
        candidate = strategy()
        if len(candidate) > len(best):
            best = set(candidate)
    
    # Try more random orderings
    for seed in range(10):
        if _time_remaining() < 10.0:
            break
        candidate = _greedy_random(seed)
        if len(candidate) > len(best):
            best = set(candidate)
    
    if not best:
        return [24]
    
    # Apply local search to the best candidate
    best = _local_search(best)
    
    # Try B&B searches from different starting points
    for seed in range(20):
        if _time_remaining() < 20.0:
            break
        candidate = _run_bbs_search(seed, max_depth=3)
        if len(candidate) > len(best):
            best = set(candidate)
    
    # Final local search polish
    best = _local_search(best)
    
    # Try a few more B&B with higher depth if time permits
    for seed in range(5):
        if _time_remaining() < 25.0:
            break
        candidate = _run_bbs_search(100 + seed, max_depth=4)
        if len(candidate) > len(best):
            best = set(candidate)
    
    return list(best)
