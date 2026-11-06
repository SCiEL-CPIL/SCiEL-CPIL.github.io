"""Rewritten program: optimized backtracking search for Sidon sets with 0,
covering differences 1..30, minimizing the maximum element."""
# EVOLVE-BLOCK-START
import time
import sys
import math

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

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
    """Improved greedy: try to cover differences with minimal max element."""
    a = [0]
    used_diffs = set()
    covered = set()
    
    for d in range(1, 31):
        if d in covered:
            continue
        found = False
        # Try adding a single element that creates difference d with some existing element
        candidates = []
        for v in a:
            for x in (v + d, v - d):
                if x >= 0 and x not in a:
                    # Check if x creates any duplicate differences
                    valid = True
                    for y in a:
                        diff = abs(x - y)
                        if diff in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
        
        candidates.sort(key=lambda x: x)  # prefer smaller values
        for x in candidates:
            # Add x and update state
            new_diffs = []
            for y in a:
                diff = abs(x - y)
                new_diffs.append(diff)
            
            a.append(x)
            a.sort()
            for diff in new_diffs:
                used_diffs.add(diff)
                if 1 <= diff <= 30:
                    covered.add(diff)
            found = True
            break
        
        if not found:
            # Need to add elements to create difference d
            # Try adding pairs (y, y+d) for increasing y
            y = max(a) + 1
            while True:
                # Check if we can add y and y+d
                valid = True
                for x in a:
                    if abs(y - x) in used_diffs or abs((y+d) - x) in used_diffs:
                        valid = False
                        break
                if abs(d) in used_diffs:
                    valid = False
                if valid:
                    a.append(y)
                    a.append(y + d)
                    a.sort()
                    for x in a:
                        for z in a:
                            if x < z:
                                used_diffs.add(z - x)
                                if 1 <= (z - x) <= 30:
                                    covered.add(z - x)
                    found = True
                    break
                y += 1
    return a

# ---------------------------------------------------------------------------
# Optimized backtracking search
# ---------------------------------------------------------------------------

def _backtracking_search(D, time_limit):
    """
    Optimized backtracking: search for Sidon set in [0, D] covering 1..30.
    Uses bitmask for covered differences and aggressive pruning.
    """
    target_mask = (1 << 30) - 1
    best_sol = [None]
    
    def dfs(current_set, covered_mask, used_diffs):
        if time.time() > time_limit:
            return False
        
        if covered_mask == target_mask:
            best_sol[0] = current_set[:]
            return True
            
        if len(current_set) >= 32:
            return False
            
        last_val = current_set[-1]
        n = len(current_set)
        
        # Quick lower bound: need at least k more elements to cover remaining diffs
        remaining = bin(target_mask ^ covered_mask).count("1")
        # Each new element can cover at most n new differences (with existing elements)
        # But we also need to be able to create all remaining diffs
        # Simple bound: if remaining > n * (32 - len(current_set)), impossible
        if remaining > n * (32 - len(current_set)):
            return False
        
        # Generate candidates efficiently: only try x that create at least one new covered diff
        candidates = []
        # Try x from last_val+1 to D, but skip many via bit operations
        for x in range(last_val + 1, D + 1):
            valid = True
            new_mask = covered_mask
            for y in current_set:
                d = x - y
                if d in used_diffs:
                    valid = False
                    break
                if 1 <= d <= 30:
                    new_mask |= (1 << (d-1))
            
            if not valid:
                continue
                
            if new_mask == covered_mask:
                continue  # No improvement in coverage
            
            improvement = bin(new_mask ^ covered_mask).count("1")
            candidates.append((improvement, x))
            
        # Sort by improvement descending, then by x ascending (prefer smaller)
        candidates.sort(key=lambda item: (-item[0], item[1]))
        
        for imp, x in candidates:
            new_set = current_set + [x]
            new_used = used_diffs.copy()
            new_mask = covered_mask
            for y in current_set:
                d = x - y
                new_used.add(d)
                if 1 <= d <= 30:
                    new_mask |= (1 << (d-1))
            
            if dfs(new_set, new_mask, new_used):
                return True
                
        return False

    if dfs([0], 0, set()):
        return best_sol[0]
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
    # Iterate D from a lower bound up to best_val
    # Use adaptive time budget: more time for smaller D values
    
    # Lower bound: need at least 10 elements for 30 differences (each pair gives one diff)
    # With 10 elements, we have C(10,2)=45 possible diffs, need 30 distinct ones in 1..30
    # Minimum diameter for 10-element Sidon set is at least ~sqrt(45)*something
    # Empirically, start from 60
    
    start_D = 55
    end_D = best_val
    
    for D in range(start_D, end_D + 1):
        elapsed = time.time() - t0
        remaining_time = _DEADLINE - time.time()
        
        if remaining_time < 3.0:
            break
            
        # Adaptive time budget: give more time to smaller D
        # For small D, search is faster; for large D, we need more time
        if D < 80:
            time_per_D = min(2.0, remaining_time / max(1, end_D - D + 1))
        elif D < 120:
            time_per_D = min(3.0, remaining_time / max(1, end_D - D + 1))
        else:
            time_per_D = min(5.0, remaining_time / max(1, end_D - D + 1))
        
        limit = time.time() + time_per_D
        
        sol = _backtracking_search(D, limit)
        
        if sol:
            # Found a solution with max element <= D
            # This is the minimum possible diameter since we iterate D increasing
            return sol
            
    # If no better solution found, return baseline
    return best_set

# EVOLVE-BLOCK-END
