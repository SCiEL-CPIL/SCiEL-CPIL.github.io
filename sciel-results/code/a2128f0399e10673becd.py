"""Optimized solver: maximize |S| <= 48 from [1..192] such that no element divides the sum of any 3 distinct OTHER elements in S."""

import time
import random
from itertools import combinations

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 85.0  # 5s safety margin before 90s kill


def _check_candidate_vs_selected(selected, candidate):
    """Check if candidate divides sum of any 3 distinct elements in selected."""
    n = len(selected)
    if n < 3:
        return True
    sel = list(selected)
    for i in range(n):
        xi = sel[i]
        for j in range(i + 1, n):
            xj = sel[j]
            target = (-xi - xj) % candidate
            for k in range(j + 1, n):
                if sel[k] % candidate == target:
                    return False
    return True


def _check_selected_vs_candidate(selected, candidate):
    """Check if any a in selected divides candidate + x + y for distinct x,y in selected\{a}."""
    sel = list(selected)
    n = len(sel)
    for a in sel:
        target = (-candidate) % a
        found = False
        for i in range(n):
            if sel[i] == a:
                continue
            needed_res = (target - sel[i]) % a
            for j in range(n):
                if sel[j] == a or sel[j] == sel[i]:
                    continue
                if sel[j] % a == needed_res:
                    found = True
                    break
            if found:
                break
        if found:
            return False
    return True


def _is_valid_addition(selected, candidate):
    """Full validity check for adding candidate to selected."""
    if not _check_candidate_vs_selected(selected, candidate):
        return False
    if not _check_selected_vs_candidate(selected, candidate):
        return False
    return True


def _full_validate(selected):
    """Full validation of the entire set."""
    sel = list(selected)
    n = len(sel)
    for idx, a in enumerate(sel):
        others = [x for i, x in enumerate(sel) if i != idx]
        if len(others) < 3:
            continue
        for x, y, z in combinations(others, 3):
            if (x + y + z) % a == 0:
                return False
    return True


def _greedy_descending():
    selected = []
    for candidate in range(192, 0, -1):
        if _is_valid_addition(selected, candidate):
            selected.append(candidate)
    return selected


def _greedy_ascending():
    selected = []
    for candidate in range(1, 193):
        if _is_valid_addition(selected, candidate):
            selected.append(candidate)
    return selected


def _random_greedy(seed_val):
    rng = random.Random(seed_val)
    candidates = list(range(1, 193))
    rng.shuffle(candidates)
    selected = []
    for c in candidates:
        if _is_valid_addition(selected, c):
            selected.append(c)
    return selected


def _local_search_add_only(initial_set, time_limit):
    """Try to add more elements to the set."""
    selected = list(initial_set)
    selected_set = set(selected)
    unselected = [x for x in range(1, 193) if x not in selected_set]
    rng = random.Random()
    
    improved = True
    while improved and (time.time() - _START_TIME) < time_limit:
        improved = False
        for c in unselected:
            if _is_valid_addition(selected, c):
                selected.append(c)
                selected_set.add(c)
                unselected.remove(c)
                improved = True
                break
    return selected


def _swap_search(initial_set, time_limit):
    """Try to improve by removing one element and adding two others."""
    selected = list(initial_set)
    selected_set = set(selected)
    unselected = [x for x in range(1, 193) if x not in selected_set]
    best_set = list(selected)
    best_len = len(selected)
    rng = random.Random()
    
    attempts = 0
    while (time.time() - _START_TIME) < time_limit and attempts < 200:
        attempts += 1
        # Pick a random element to remove
        idx = rng.randint(0, len(selected) - 1)
        removed = selected.pop(idx)
        selected_set.remove(removed)
        
        # Try to add two elements from unselected + removed
        add_pool = list(unselected) + [removed]
        rng.shuffle(add_pool)
        added_count = 0
        for c in add_pool:
            if added_count >= 2:
                break
            if _is_valid_addition(selected, c):
                selected.append(c)
                selected_set.add(c)
                added_count += 1
        
        if len(selected) > best_len:
            best_len = len(selected)
            best_set = list(selected)
            unselected = [x for x in range(1, 193) if x not in selected_set]
        else:
            # Revert
            for c in selected:
                if c not in best_set and c != removed:
                    pass
            # Simpler: just revert by restoring removed
            # Actually let's just revert properly
            # This is getting complex, let's use a different approach
        
    return best_set


def _multi_start_local_search(initial_set, time_limit):
    """Enhanced local search with perturbations."""
    selected = list(initial_set)
    selected_set = set(selected)
    best_set = list(selected)
    best_len = len(selected)
    rng = random.Random()
    
    while (time.time() - _START_TIME) < time_limit:
        # First try to add
        unselected = [x for x in range(1, 193) if x not in selected_set]
        added_any = False
        for c in unselected:
            if _is_valid_addition(selected, c):
                selected.append(c)
                selected_set.add(c)
                added_any = True
                break
        
        if added_any:
            if len(selected) > best_len:
                best_len = len(selected)
                best_set = list(selected)
            continue
        
        # Can't add more, try perturbation: remove 1 random, then greedy add
        if len(selected) > 0:
            idx = rng.randint(0, len(selected) - 1)
            removed = selected.pop(idx)
            selected_set.remove(removed)
            
            # Greedily add back from unselected
            unselected2 = [x for x in range(1, 193) if x not in selected_set]
            rng.shuffle(unselected2)
            for c in unselected2:
                if (time.time() - _START_TIME) >= time_limit:
                    break
                if _is_valid_addition(selected, c):
                    selected.append(c)
                    selected_set.add(c)
            
            if len(selected) > best_len:
                best_len = len(selected)
                best_set = list(selected)
            # Don't revert, continue from current state for diversity
    
    return best_set


def construct():
    start_time = time.time()
    best_set = []
    best_len = 0
    
    def try_update(s):
        nonlocal best_set, best_len
        if len(s) > best_len:
            best_set = list(s)
            best_len = len(s)
    
    # Phase 1: Deterministic greedy approaches
    for strat_fn in [_greedy_descending, _greedy_ascending]:
        if (time.time() - start_time) > _DEADLINE - 5:
            break
        s = strat_fn()
        try_update(s)
        # Quick local search improvement
        if (time.time() - start_time) < _DEADLINE - 3:
            s2 = _local_search_add_only(s, _DEADLINE - 2)
            try_update(s2)
    
    # Phase 2: Random restarts
    num_restarts = 30
    for i in range(num_restarts):
        if (time.time() - start_time) > _DEADLINE - 3:
            break
        s = _random_greedy(i)
        try_update(s)
        if (time.time() - start_time) < _DEADLINE - 2:
            s2 = _local_search_add_only(s, _DEADLINE - 1.5)
            try_update(s2)
    
    # Phase 3: Deeper local search on best found so far
    if (time.time() - start_time) < _DEADLINE - 1 and len(best_set) > 0:
        remaining = _DEADLINE - 1 - (time.time() - start_time)
        if remaining > 1.0:
            improved = _multi_start_local_search(best_set, _DEADLINE - 0.5)
            try_update(improved)
    
    # Final validation
    best_set = list(set(best_set))
    best_set = [x for x in best_set if 1 <= x <= 192]
    
    # Ensure validity
    if best_set and not _full_validate(best_set):
        # Fallback: rebuild greedily
        best_set = _greedy_descending()
        if not _full_validate(best_set):
            best_set = []
    
    return best_set


# EVOLVE-BLOCK-END
