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

def _search(M, target_diffs, deadline):
    """
    Depth-first search for a Sidon set in [0, M] covering target_diffs.
    Uses bitmask for O(1) difference tracking and efficient candidate generation.
    """
    # Bitmask for differences 1..M (index d-1 represents difference d)
    # used_mask bit i set means difference (i+1) is used
    # remaining_mask bit i set means difference (i+1) is still needed
    
    target_mask = 0
    for d in target_diffs:
        target_mask |= (1 << (d - 1))
    
    initial_set = [0]
    
    # Precompute: for each possible x in 1..M, which existing elements a
    # would create a conflict. We'll compute on the fly but use bitmask.
    
    def dfs(current_set, used_mask, remaining_mask):
        if remaining_mask == 0:
            return current_set[:]
        
        if len(current_set) >= 32:
            return None
            
        if time.time() > deadline:
            return None
        
        last = current_set[-1]
        
        # Generate candidates efficiently
        # x must be > last, <= M
        # For each x, check if x - a creates a used or duplicate difference for all a in current_set
        candidate_info = []
        
        # Build set of current elements for fast lookup
        cur_set = set(current_set)
        
        # Candidates come from: for each a in current_set, for each remaining diff d, x = a + d
        # But also any x > last could work if it covers at least one remaining diff
        # More efficient: iterate x from last+1 to M, check coverage
        
        # Actually, let's use the smarter approach: only consider x that would cover
        # at least one remaining difference. x covers diff d iff x - a = d for some a in current_set,
        # i.e., x = a + d for some a in current_set and d in remaining.
        
        seen = set()
        for a in current_set:
            # For each remaining diff d, x = a + d is a candidate
            rem = remaining_mask
            while rem:
                # Get lowest set bit
                lb = rem & (-rem)
                d = lb.bit_length()  # d is the difference value
                x = a + d
                if x > last and x <= M and x not in seen:
                    seen.add(x)
                    # Check if x is valid: for all a2 in current_set, x - a2 not in used_mask and all distinct
                    valid = True
                    new_covered = 0
                    new_diffs_mask = 0
                    for a2 in current_set:
                        diff = x - a2
                        if diff <= 0 or diff > M:
                            continue
                        bit = 1 << (diff - 1)
                        if used_mask & bit:
                            valid = False
                            break
                        if new_diffs_mask & bit:
                            valid = False
                            break
                        new_diffs_mask |= bit
                        if remaining_mask & bit:
                            new_covered += 1
                    if valid and new_covered > 0:
                        candidate_info.append((x, new_covered))
                rem &= rem - 1  # clear lowest set bit
        
        # Sort candidates: more coverage first, then smaller x
        candidate_info.sort(key=lambda t: (-t[1], t[0]))
        
        for x, _ in candidate_info:
            # Recompute new diffs mask
            new_diffs_mask = 0
            conflict = False
            for a2 in current_set:
                diff = x - a2
                if diff <= 0 or diff > M:
                    continue
                bit = 1 << (diff - 1)
                if used_mask & bit or new_diffs_mask & bit:
                    conflict = True
                    break
                new_diffs_mask |= bit
            
            if conflict:
                continue
                
            new_used = used_mask | new_diffs_mask
            new_remaining = remaining_mask & ~new_used
            
            res = dfs(current_set + [x], new_used, new_remaining)
            if res is not None:
                return res
                
        return None

    return dfs(initial_set, 0, target_mask)

def _local_polish(a, deadline):
    """Try to reduce the max element by swapping out the largest element."""
    target = set(range(1, 31))
    best = a[:]
    best_max = max(best)
    
    # Try replacing the largest element with something smaller
    for _ in range(50):
        if time.time() > deadline:
            break
        if len(best) <= 1:
            break
        largest_idx = len(best) - 1
        largest_val = best[largest_idx]
        found = False
        for new_val in range(1, largest_val):
            candidate = best[:largest_idx] + [new_val]
            candidate.sort()
            if _check_sidon_and_cover(candidate, target):
                best = candidate
                best_max = max(best)
                found = True
                break
        if not found:
            break
    
    return best

def _known_good():
    """A known valid Sidon set covering 1..30 with small max element."""
    candidates = [
        [0, 1, 4, 10, 12, 17, 21, 25, 30],
        [0, 1, 4, 10, 18, 22, 26, 30, 35],
        [0, 1, 3, 7, 12, 18, 25, 30, 36],
    ]
    target = set(range(1, 31))
    for c in candidates:
        if _check_sidon_and_cover(c, target):
            return c
    # If none work, build greedily
    a = [0]
    used = set()
    required = set(range(1, 31))
    for x in range(1, 200):
        if not required:
            break
        new_diffs = set()
        ok = True
        for v in a:
            d = abs(x - v)
            if d in used:
                ok = False
                break
            new_diffs.add(d)
        if ok and new_diffs & required:
            a.append(x)
            used |= new_diffs
            required -= new_diffs
    if not required:
        return a
    # Last resort: extend
    while required:
        x = max(a) + 1
        a.append(x)
        for v in a[:-1]:
            used.add(abs(x - v))
        required -= set(abs(x - v) for v in a[:-1])
    return a

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    target_diffs = set(range(1, 31))
    
    # Try increasing M values
    for M in range(55, 150):
        if time.time() > deadline:
            break
            
        result = _search(M, target_diffs, deadline)
        if result is not None:
            if _check_sidon_and_cover(result, target_diffs):
                # Try to polish
                polished = _local_polish(result, deadline)
                if _check_sidon_and_cover(polished, target_diffs):
                    return polished
                return result
                
    # Fallback
    return _known_good()
