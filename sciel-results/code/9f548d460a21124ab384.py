"""Improved solver for B2[2] sets in 1..1000.

Strategy:
- Fast greedy baseline to get a valid set quickly.
- Local search with phased approach: first try additions, then targeted swaps.
- Efficient swap check with temporary removal and early break.
- Deadline guard to ensure we return within the time limit.
"""

import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    N = 1000
    MAX_REP = 2

    def _time_left():
        return deadline - time.time()

    def _can_add(arr, x, counts):
        """Check if adding x to arr keeps all counts <= MAX_REP."""
        if x in arr:
            return False
        for a in arr:
            s = x + a
            if counts[s] >= MAX_REP:
                return False
        s2 = 2 * x
        if counts[s2] >= MAX_REP:
            return False
        return True

    def _add_element(arr, x, counts):
        """Add x to arr and update counts."""
        arr.append(x)
        for a in arr[:-1]:
            counts[x + a] += 1
        counts[2 * x] += 1

    def _remove_element(arr, x, counts):
        """Remove x from arr and update counts."""
        arr.remove(x)
        for a in arr:
            counts[x + a] -= 1
        counts[2 * x] -= 1

    # Phase 1: Fast greedy
    arr = []
    counts = defaultdict(int)
    for x in range(1, N + 1):
        if _time_left() < 0:
            break
        ok = True
        for a in arr:
            s = x + a
            if counts[s] >= MAX_REP:
                ok = False
                break
        if ok and counts[2 * x] >= MAX_REP:
            ok = False
        if ok:
            _add_element(arr, x, counts)

    arr_set = set(arr)
    missing = [x for x in range(1, N + 1) if x not in arr_set]
    
    # Phase 2: Local search with phased additions then swaps
    improved = True
    while improved and _time_left() > 1.0:
        improved = False
        
        # Try to add any missing element
        for x in missing[:]:
            if _time_left() < 0.5:
                break
            if _can_add(arr, x, counts):
                _add_element(arr, x, counts)
                arr_set.add(x)
                missing.remove(x)
                improved = True
                break
        if improved:
            continue
        
        # Try swaps: remove one element, add another
        if not missing:
            break
            
        # Try targeted swaps with early break
        max_attempts = 50
        for attempt in range(max_attempts):
            if _time_left() < 0.5:
                break
            if not missing:
                break
                
            # Pick a random element to remove
            idx = random.randrange(len(arr))
            removed = arr[idx]
            
            # Try to find a missing element that can replace 'removed'
            found = False
            for x in missing:
                if _time_left() < 0.5:
                    break
                    
                # Temporarily remove 'removed'
                _remove_element(arr, removed, counts)
                
                # Check if x can be added
                if _can_add(arr, x, counts):
                    # Accept the swap
                    _add_element(arr, x, counts)
                    arr_set.discard(removed)
                    arr_set.add(x)
                    missing.remove(x)
                    found = True
                    improved = True
                    break
                else:
                    # Restore 'removed'
                    _add_element(arr, removed, counts)
                    continue
            
            # If we broke out of inner loop without finding, restore if needed
            if not found and removed not in arr:
                _add_element(arr, removed, counts)
                arr_set.add(removed)
    
    return arr


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"First 20: {result[:20]}")
