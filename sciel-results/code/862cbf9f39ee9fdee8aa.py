"""Rewritten program: optimized backtracking search for Sidon sets with 0,
covering differences 1..30, minimizing the maximum element."""
# EVOLVE-BLOCK-START
import time
import sys
import math

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill

# ---------------------------------------------------------------------------
# Fast helpers
# ---------------------------------------------------------------------------

def _is_sidon(s):
    """True iff all positive differences are distinct."""
    n = len(s)
    seen = set()
    for i in range(n):
        si = s[i]
        for j in range(i + 1, n):
            d = s[j] - si
            if d in seen:
                return False
            seen.add(d)
    return True

def _covers_1_to_30(s):
    d = set()
    for i in range(len(s)):
        si = s[i]
        for j in range(i + 1, len(s)):
            d.add(s[j] - si)
    for x in range(1, 31):
        if x not in d:
            return False
    return True

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

# ---------------------------------------------------------------------------
# Greedy baseline (fast, deterministic)
# ---------------------------------------------------------------------------

def _greedy_construct():
    a = [0]
    def diffs_list(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs_list(sorted(a)):
            continue
        found = False
        candidates = []
        for v in a:
            for x in (v + d, v - d):
                if x >= 0 and x not in a:
                    candidates.append(x)
        candidates = sorted(set(candidates))
        for x in candidates:
            trial = sorted(a + [x])
            dl = diffs_list(trial)
            if len(set(dl)) == len(dl):
                a = trial
                found = True
                break
        if not found:
            y = max(a) + 1
            while True:
                trial = sorted(a + [y, y + d])
                dl = diffs_list(trial)
                if len(set(dl)) == len(dl):
                    a = trial
                    break
                y += 1
    return a

# ---------------------------------------------------------------------------
# Optimized backtracking search
# ---------------------------------------------------------------------------

def _backtracking_search(D, time_limit):
    """
    Search for a valid Sidon set in [0, D] covering 1..30.
    Returns the set if found, None otherwise.
    """
    target_mask = (1 << 30) - 1
    best_sol = None
    
    # Precompute for each possible x > 0, which differences it creates with elements 0..x-1
    # This is used for fast candidate generation
    
    def dfs(current_set, covered_mask, used_diffs):
        nonlocal best_sol
        if time.time() > time_limit:
            return False
        
        if covered_mask == target_mask:
            best_sol = current_set[:]
            return True
            
        if len(current_set) >= 32:
            return False
            
        last_val = current_set[-1]
        
        # Generate candidates x > last_val
        # Sort by improvement in coverage
        candidates = []
        for x in range(last_val + 1, D + 1):
            valid = True
            new_diffs = []
            for y in current_set:
                d = x - y
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << (d-1))
                    
            improvement = bin(new_mask ^ covered_mask).count("1")
            if improvement == 0:
                continue
                
            candidates.append((improvement, x, new_diffs, new_mask))
            
        candidates.sort(key=lambda item: item[0], reverse=True)
        
        for imp, x, new_diffs, new_mask in candidates:
            new_set = current_set + [x]
            new_used = used_diffs.union(new_diffs)
            
            if dfs(new_set, new_mask, new_used):
                return True
                
        return False

    if dfs([0], 0, set()):
        return best_sol
    return None

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    t0 = time.time()
    
    # Get a baseline solution to ensure we have something valid
    baseline = _greedy_construct()
    best_val = max(baseline)
    best_set = baseline
    
    # Try to find a smaller diameter using backtracking
    # Start from a lower bound.
    # We need at least 9 elements.
    # A safe lower bound for diameter is 50.
    # We iterate D upwards.
    
    # To save time, we can skip D values if we are close to deadline
    # But backtracking is fast for small D.
    
    # Let's try D from 80 to best_val
    # If we find a solution for D, it's the minimum for that D?
    # Yes, because we search subsets of [0, D].
    
    # However, if we find a solution for D=100, it might have max element 90.
    # So the true diameter is <= 100.
    # We want to minimize the max element.
    # So we should return the actual max element of the found set.
    
    # We iterate D. The first D that yields a solution gives us a set with max <= D.
    # Since we iterate D increasing, the first solution found will have the smallest possible D_upper.
    # But the actual max element might be smaller than D.
    # Is it possible that a solution with max element 95 exists, but we first find one with max element 100 at D=100?
    # No, if a solution with max 95 exists, it is a subset of [0, 95].
    # So it would be found when D=95.
    # So the first D that yields a solution corresponds to the minimum possible diameter.
    
    # We start from a lower bound.
    # Let's estimate lower bound.
    # If baseline is 200, we can start from 100.
    # But to be safe, start from 50.
    
    start_D = 50
    # We don't need to go above best_val
    end_D = best_val
    
    for D in range(start_D, end_D + 1):
        if time.time() > _DEADLINE - 5:
            break
            
        # Allow 5 seconds for each D
        t_start = time.time()
        limit = t_start + 5.0
        
        sol = _backtracking_search(D, limit)
        
        if sol:
            # This is the minimum diameter found so far
            # We can return immediately because we are iterating D upwards
            # and any solution in [0, D] has diameter <= D.
            # Since no solution was found for D-1, this is the minimum.
            return sol
            
    # If no better solution found, return baseline
    return best_set

# EVOLVE-BLOCK-END
