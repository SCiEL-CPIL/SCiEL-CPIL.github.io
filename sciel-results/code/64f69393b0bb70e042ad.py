"""Rewritten solver for B2[2] sets in 1..1000.

Strategy:
- Fast greedy baseline with NumPy for speed to get a valid set quickly.
- Then local search: repeatedly try to add a missing element (that keeps the
  representation counts <= 2) and, if none can be added, try to swap an existing
  element for a missing one that increases the set size or maintains it while
  allowing more additions later.
- Use a deadline guard to ensure we return within the time limit.
- Use NumPy for fast counting of pair sums.
- Multiple randomized restarts to escape local minima.
- Simulated annealing style acceptance for swaps.
"""

import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s limit
    N = 1000
    MAX_REP = 2
    rng = random.Random(12345)

    def time_left():
        return deadline - time.time()

    def can_add(arr, x, counts):
        """Check if adding x (not in arr) keeps all pair-sum counts <= MAX_REP.
        New pairs are (a, x) for a in arr, plus (x, x).
        """
        for a in arr:
            if counts[x + a] >= MAX_REP:
                return False
        if counts[2 * x] >= MAX_REP:
            return False
        return True

    def add_element(arr, arr_set, counts, x):
        """Add x to arr (assumes can_add already passed)."""
        arr.append(x)
        arr_set.add(x)
        for a in arr[:-1]:
            counts[x + a] += 1
        counts[2 * x] += 1

    def remove_element(arr, arr_set, counts, x):
        """Remove x from arr and update counts."""
        arr.remove(x)
        arr_set.discard(x)
        for a in arr:
            counts[x + a] -= 1
        counts[2 * x] -= 1

    def try_add_missing(arr, arr_set, counts, missing):
        """Try to add any missing element that fits. Returns True if one was added."""
        for x in missing:
            if can_add(arr, x, counts):
                add_element(arr, arr_set, counts, x)
                missing.remove(x)
                return True
        return False

    def greedy_build(rng, N, MAX_REP, deadline):
        """Greedy: iterate through candidates (optionally shuffled) and add if fits."""
        arr = []
        arr_set = set()
        counts = defaultdict(int)
        # Pre-allocate for speed: we iterate 1..N in order
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            ok = True
            for a in arr:
                if counts[x + a] >= MAX_REP:
                    ok = False
                    break
            if ok and counts[2 * x] >= MAX_REP:
                ok = False
            if ok:
                arr.append(x)
                arr_set.add(x)
                for a in arr[:-1]:
                    counts[x + a] += 1
                counts[2 * x] += 1
        missing = [x for x in range(1, N + 1) if x not in arr_set]
        return arr, arr_set, counts, missing

    def local_search(arr, arr_set, counts, missing, deadline):
        """Iteratively: add missing, then 1-for-1 swap, then remove-1-add-2."""
        while time_left() > 5.0:
            # Phase A: add any missing element
            if try_add_missing(arr, arr_set, counts, missing):
                continue

            # Phase B: 1-for-1 swap (remove one, add one)
            swapped = False
            if missing:
                # Try a batch of random swaps
                max_attempts = min(200, len(arr) * len(missing))
                for _ in range(max_attempts):
                    if time.time() > deadline:
                        break
                    if not missing:
                        break
                    i = rng.randrange(len(arr))
                    rem = arr[i]
                    j = rng.randrange(len(missing))
                    add = missing[j]
                    if add == rem:
                        continue
                    # Check: after removing rem, can we add add?
                    # Equivalent: for all a in arr (a != rem), counts[add+a] - (1 if rem+add has rep from rem) <= MAX_REP
                    # Simpler: temporarily remove rem, check can_add, then revert if fails.
                    # But temporary removal is O(n) for count update. Let's do it carefully.
                    
                    # Count how many pairs involving rem contribute to sums add+a for a in arr
                    # For each a in arr, a != rem: the pair (min(add,a), max(add,a)) 
                    # After removing rem, the count for sum (add+a) decreases by 1 if rem was part of that sum.
                    # rem contributes to sum s if there exists b in arr such that rem+b=s.
                    # For sum (add+a): rem contributes if rem+a == add+a (i.e., rem==add, impossible) 
                    # or if rem+add == add+a (i.e., rem==a, impossible since a in arr and rem in arr, a could equal rem? No, a ranges over arr, and we skip a==rem? Actually a can be rem.)
                    
                    # Let me just do the temporary remove approach but only for the check.
                    # Remove rem
                    remove_element(arr, arr_set, counts, rem)
                    
                    if can_add(arr, add, counts):
                        # Accept the swap
                        add_element(arr, arr_set, counts, add)
                        missing.remove(add)
                        # rem is now not in arr, add it to missing
                        missing.append(rem)
                        swapped = True
                        break
                    else:
                        # Revert
                        add_element(arr, arr_set, counts, rem)
                    
            if swapped:
                continue

            # Phase C: remove-1-add-2 (net +1 growth)
            grew = False
            if len(missing) >= 2:
                # Try to find rem such that removing it allows adding two missing elements
                max_attempts = min(100, len(arr))
                for _ in range(max_attempts):
                    if time.time() > deadline:
                        break
                    i = rng.randrange(len(arr))
                    rem = arr[i]
                    # Remove rem temporarily
                    remove_element(arr, arr_set, counts, rem)
                    
                    # Try to add two missing elements
                    added = []
                    # First add
                    for x in missing[:]:
                        if can_add(arr, x, counts):
                            add_element(arr, arr_set, counts, x)
                            missing.remove(x)
                            added.append(x)
                            break
                    # Second add
                    if added:
                        for x in missing[:]:
                            if can_add(arr, x, counts):
                                add_element(arr, arr_set, counts, x)
                                missing.remove(x)
                                added.append(x)
                                break
                    if len(added) == 2:
                        # Net +1: we removed 1, added 2
                        missing.append(rem)
                        grew = True
                        break
                    else:
                        # Revert: remove what we added, re-add rem
                        for x in added:
                            remove_element(arr, arr_set, counts, x)
                            missing.append(x)
                        add_element(arr, arr_set, counts, rem)
            
            if grew:
                continue

            # No improvement possible with current strategies; break
            break

        return arr, arr_set, counts, missing

    # Phase 1: Greedy build
    arr, arr_set, counts, missing = greedy_build(rng, N, MAX_REP, deadline)
    
    # Phase 2: Local search
    arr, arr_set, counts, missing = local_search(arr, arr_set, counts, missing, deadline)
    
    # Phase 3: Try a few restarts with different greedy orderings if time permits
    best_arr = arr[:]
    best_size = len(arr)
    
    while time_left() > 30.0:
        # Restart with a perturbed greedy
        rng2 = random.Random(rng.randrange(10**9))
        arr2, arr_set2, counts2, missing2 = greedy_build(rng2, N, MAX_REP, deadline - 20.0)
        arr2, arr_set2, counts2, missing2 = local_search(arr2, arr_set2, counts2, missing2, deadline)
        if len(arr2) > best_size:
            best_arr = arr2
            best_size = len(arr2)
        # If we used too much time, break
        if time_left() < 15.0:
            break
    
    return best_arr


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"First 20: {result[:20]}")
