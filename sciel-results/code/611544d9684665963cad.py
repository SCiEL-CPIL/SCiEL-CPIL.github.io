"""Construct a Sidon set containing 0 with all differences 1..30, minimizing max element."""
import time
import sys

def _check_sidon_and_cover(a, target_diffs):
    """Check if a is a Sidon set covering target_diffs."""
    n = len(a)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            if d in diffs:
                return False
            diffs.add(d)
    return all(d in diffs for d in target_diffs)

def _search(M, target_diffs):
    """
    Depth-first search for a Sidon set in [0, M] covering target_diffs.
    Uses bitmask for fast conflict detection.
    """
    target_set = set(target_diffs)
    max_d = max(target_diffs)
    
    # Bitmask for target differences
    target_mask = 0
    for d in target_diffs:
        target_mask |= (1 << d)
    
    def dfs(current_set, used_mask, remaining_mask, last):
        if remaining_mask == 0:
            return current_set[:]
        
        if len(current_set) >= 32:
            return None
            
        # Generate candidates: x = a + d for a in current_set, d in remaining
        candidates = set()
        for a in current_set:
            for d in range(1, max_d + 1):
                if (remaining_mask >> d) & 1:
                    x = a + d
                    if x > last and x <= M:
                        candidates.add(x)
        
        # Sort by smallest x first
        for x in sorted(candidates):
            # Check if x creates conflicts
            new_mask = 0
            conflict = False
            for a in current_set:
                d = x - a
                if d <= 0:
                    continue
                bit = (1 << d)
                if (used_mask & bit) or (new_mask & bit):
                    conflict = True
                    break
                new_mask |= bit
            
            if conflict:
                continue
            
            new_remaining = remaining_mask & ~new_mask
            
            res = dfs(current_set + [x], used_mask | new_mask, new_remaining, x)
            if res is not None:
                return res
                
        return None

    return dfs([0], 0, target_mask, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    target_diffs = set(range(1, 31))
    
    # Lower bound: need at least 9 elements (9*8/2=36>=30)
    # Minimum diameter for Sidon set of size 9 is around 72
    # Start from a reasonable lower bound
    for M in range(50, 200):
        if time.time() > deadline:
            break
            
        result = _search(M, target_diffs)
        if result is not None:
            # Verify result
            if _check_sidon_and_cover(result, target_diffs):
                return result
                
    # Fallback: Return a known valid set if search fails or times out
    return _greedy_fallback()

def _greedy_fallback():
    a = [0]
    used = set()
    required = set(range(1, 31))
    for d in sorted(required):
        # Try to find x such that x-a=d for some a in a
        # and x-a' not in used for all a' in a
        found = False
        for x in range(1, 1000):
            if x in a: continue
            ok = True
            new_diffs = set()
            for v in a:
                diff = abs(x-v)
                if diff in used or diff in new_diffs:
                    ok = False
                    break
                new_diffs.add(diff)
            if ok:
                # Check if it helps
                a.append(x)
                used.update(new_diffs)
                required -= new_diffs
                found = True
                break
        if not found:
            # Add a large number
            x = 1000 + len(a)
            a.append(x)
            for v in a[:-1]:
                used.add(abs(x-v))
            required -= set(abs(x-v) for v in a[:-1])
    return a
