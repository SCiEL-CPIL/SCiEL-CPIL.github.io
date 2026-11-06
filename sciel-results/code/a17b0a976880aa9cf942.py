"""Optimized solver for: no element divides sum of any 3 distinct other selected elements."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
import random

MAX_N = 192
MAX_K = 48
TIME_LIMIT = 85.0  # safety margin before 90s kill

def _is_valid_set(selected):
    """Check if selected set satisfies the constraint."""
    n = len(selected)
    if n == 0:
        return True
    sel = list(selected)
    # For each element a, check that no 3 distinct OTHER elements sum to a multiple of a
    for i, a in enumerate(sel):
        others = [v for j, v in enumerate(sel) if j != i]
        # If fewer than 3 others, constraint is vacuously true for this a
        if len(others) < 3:
            continue
        # Check all triples of others
        for triple in combinations(others, 3):
            s = triple[0] + triple[1] + triple[2]
            if s % a == 0:
                return False
    return True

def _can_add(selected, x):
    """Check if adding x to selected keeps the set valid."""
    # Check: for each existing element a, does x create a bad triple with a?
    # A bad triple for a means: x + b + c where b,c are in selected, b != c, b != x, c != x
    #   and (x + b + c) % a == 0
    # Also: for x itself, does any triple of existing elements sum to multiple of x?
    
    sel = list(selected)
    n = len(sel)
    
    # Check x as the divisor: any 3 elements from sel sum to multiple of x?
    if n >= 3:
        for triple in combinations(sel, 3):
            if (triple[0] + triple[1] + triple[2]) % x == 0:
                return False
    
    # Check each existing a as divisor: does x + b + c (b,c in sel, b!=c) give (x+b+c)%a==0?
    # Note: b and c must be distinct elements from sel (and distinct from x, which they are since x is new)
    if n >= 2:
        for a in sel:
            # We need b, c from sel, b != c, such that (x + b + c) % a == 0
            # Precompute: for each pair (b,c) in sel, check if (x + b + c) % a == 0
            for i in range(n):
                b = sel[i]
                for j in range(i + 1, n):
                    c = sel[j]
                    if (x + b + c) % a == 0:
                        return False
    elif n == 1:
        # Only one existing element a; need 2 others for a triple with x, but only 1 other exists
        # So no triple can be formed involving x and 2 others from sel
        pass
    
    return True

def _greedy_descending():
    """Greedy: add numbers from high to low if valid."""
    selected = []
    for x in range(MAX_N, 0, -1):
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _greedy_ascending():
    """Greedy: add numbers from low to high if valid."""
    selected = []
    for x in range(1, MAX_N + 1):
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _local_search(selected, start_time):
    """Try to improve via swaps and additions."""
    best = list(selected)
    best_size = len(best)
    deadline = start_time + TIME_LIMIT - 5.0
    
    while time.time() < deadline:
        improved = False
        # Try adding a new element
        current_set = set(best)
        for x in range(1, MAX_N + 1):
            if x in current_set:
                continue
            if _can_add(best, x):
                new_set = best + [x]
                if _is_valid_set(new_set):
                    best = new_set
                    best_size += 1
                    current_set = set(best)
                    improved = True
                    break
        if improved:
            continue
        
        # Try swaps: remove one, add another
        for i in range(len(best)):
            if time.time() >= deadline:
                break
            removed = best[i]
            reduced = best[:i] + best[i+1:]
            for x in range(1, MAX_N + 1):
                if x in set(reduced) or x == removed:
                    continue
                if _can_add(reduced, x):
                    new_set = reduced + [x]
                    if _is_valid_set(new_set):
                        best = new_set
                        current_set = set(best)
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            break
    
    return best

def _randomized_greedy(start_time):
    """Randomized greedy with random ordering."""
    deadline = start_time + TIME_LIMIT - 10.0
    best = []
    best_size = 0
    attempts = 0
    
    while time.time() < deadline:
        attempts += 1
        # Random ordering
        candidates = list(range(1, MAX_N + 1))
        random.shuffle(candidates)
        
        selected = []
        for x in candidates:
            if len(selected) >= MAX_K:
                break
            if _can_add(selected, x):
                selected.append(x)
        
        if len(selected) > best_size:
            best = selected
            best_size = len(best)
    
    return best

def construct():
    start_time = time.time()
    
    # Phase 1: Deterministic greedy (fast)
    result = _greedy_descending()
    if not _is_valid_set(result):
        # Fallback: rebuild carefully
        result = []
        for x in range(MAX_N, 0, -1):
            if _can_add(result, x):
                result.append(x)
    
    # Phase 2: Local search to improve
    result = _local_search(result, start_time)
    
    # Phase 3: Randomized greedy if time permits
    if time.time() < start_time + TIME_LIMIT - 15.0:
        rand_result = _randomized_greedy(start_time)
        if len(rand_result) > len(result):
            result = rand_result
        else:
            # Try local search on randomized result too
            rand_improved = _local_search(rand_result, start_time)
            if len(rand_improved) > len(result):
                result = rand_improved
    
    # Final validity check
    if not _is_valid_set(result):
        # Rebuild from scratch with careful validation
        result = []
        for x in range(MAX_N, 0, -1):
            if _can_add(result, x):
                result.append(x)
    
    return result
# EVOLVE-BLOCK-END
