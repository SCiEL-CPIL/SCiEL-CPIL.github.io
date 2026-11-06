# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _positive_diffs(s):
    n = len(s)
    return [s[j] - s[i] for i in range(n) for j in range(i + 1, n)]

def _is_sidon(s):
    d = _positive_diffs(s)
    return len(d) == len(set(d))

def _covers_1_to_30(s):
    d = set(_positive_diffs(s))
    return all(x in d for x in range(1, 31))

def _valid(s):
    if len(s) < 1 or len(s) > 32:
        return False
    if 0 not in s:
        return False
    if max(s) > 10**6:
        return False
    if not _is_sidon(s):
        return False
    if not _covers_1_to_30(s):
        return False
    return True

def _backtracking_search():
    """
    Backtracking search with heuristic ordering.
    We build the set incrementally, always trying to cover the smallest uncovered difference.
    """
    best = [0, 1, 2, 4, 7, 13, 21, 31]  # initial guess, will be improved
    best_val = max(best) if best else 10**9
    
    # Try to find a better solution using backtracking
    # Start with 0 and try to add elements that cover the most needed differences
    
    needed = set(range(1, 31))
    
    def solve(current, remaining_needed, depth):
        nonlocal best, best_val
        if time.time() > _DEADLINE:
            return False
        if not remaining_needed:
            if len(current) <= 32 and max(current) < best_val:
                best = current[:]
                best_val = max(current)
            return True
        
        if len(current) >= 32:
            return False
            
        # Find the smallest needed difference
        target = min(remaining_needed)
        
        # Try adding elements that could create this difference
        # If we add x, then x - v = target or v - x = target for some v in current
        candidates = set()
        for v in current:
            candidates.add(v + target)
            if v - target > 0:
                candidates.add(v - target)
        
        # Sort candidates by how many needed differences they'd cover
        def score_candidate(x):
            if x <= 0 or x in current:
                return -1
            trial = sorted(current + [x])
            if len(trial) != len(set(trial)):
                return -1
            diffs = set(trial[j] - trial[i] for i in range(len(trial)) for j in range(i+1, len(trial)))
            if len(diffs) != len(_positive_diffs(trial)):
                return -1  # not Sidon
            covered = len(diffs & remaining_needed)
            return covered
        
        scored = [(score_candidate(x), x) for x in candidates]
        scored = [(s, x) for s, x in scored if s >= 0]
        scored.sort(key=lambda t: (-t[0], t[1]))
        
        for score, x in scored:
            if time.time() > _DEADLINE:
                return False
            trial = sorted(current + [x])
            # Check Sidon
            diffs = _positive_diffs(trial)
            if len(set(diffs)) != len(diffs):
                continue
            # Update remaining needed
            diff_set = set(diffs)
            new_remaining = remaining_needed - diff_set
            if solve(trial, new_remaining, depth + 1):
                return True
        
        # If no single element works, try adding two elements
        # This is more expensive, so only do it if we haven't found a solution yet
        if depth < 10 and not remaining_needed <= set(range(1, 31)):
            # Try adding pairs
            max_try = 200
            base = max(current) + 1
            for offset in range(max_try):
                if time.time() > _DEADLINE:
                    return False
                y = base + offset
                z = y + target
                if z > 10**6:
                    break
                trial = sorted(current + [y, z])
                if len(trial) != len(set(trial)):
                    continue
                diffs = _positive_diffs(trial)
                if len(set(diffs)) != len(diffs):
                    continue
                diff_set = set(diffs)
                new_remaining = remaining_needed - diff_set
                if solve(trial, new_remaining, depth + 1):
                    return True
        
        return False
    
    solve([0], needed, 0)
    return best

def _local_search(start_set, max_steps=50000):
    """
    Local search to minimize max element.
    """
    best = start_set[:]
    best_val = max(start_set)
    rng = random.Random(42)
    current = start_set[:]
    current_val = max(current)
    
    steps = 0
    while time.time() < _DEADLINE and steps < max_steps:
        steps += 1
        # Pick a random non-zero element to move
        idx = rng.randint(1, len(current) - 1)
        old_val = current[idx]
        
        # Try moving it to a nearby value
        delta = rng.randint(-30, 10)
        new_val = old_val + delta
        
        if new_val <= 0:
            continue
            
        if new_val in current and new_val != old_val:
            continue
            
        candidate = current[:]
        candidate[idx] = new_val
        candidate.sort()
        
        if _is_sidon(candidate) and _covers_1_to_30(candidate):
            val = max(candidate)
            if val < current_val:
                current = candidate
                current_val = val
                if val < best_val:
                    best_val = val
                    best = current[:]
    
    return best

def construct():
    t0 = time.time()
    
    # 1. Backtracking search
    bt = _backtracking_search()
    best = bt
    best_val = max(best)
    
    # 2. Local search from backtracking result
    if time.time() < _DEADLINE:
        ls = _local_search(best, max_steps=30000)
        if ls is not None and max(ls) < best_val:
            best = ls
            best_val = max(ls)
    
    return best
# EVOLVE-BLOCK-END
