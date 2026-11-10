"""Improved program for the no-divide-sum-of-three-others task."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _is_valid(selected):
    """Check if no element in selected divides the sum of any 3 distinct other selected elements."""
    s = set(selected)
    k = len(s)
    if k < 4:
        return True
    lst = list(s)
    for a in lst:
        others = [x for x in lst if x != a]
        # Check if there exist 3 distinct elements in others whose sum is divisible by a
        # Use: for each pair (i,j) in others, check if -(others[i]+others[j]) mod a is in others (as a value, distinct from i,j)
        # Build a dict: value -> list of indices, or just a set of values
        others_set = set(others)
        # For efficiency, group others by value mod a
        # For each pair, compute needed remainder
        n = len(others)
        # O(n^2) check: for each pair (x,y), check if there exists z in others, z != x, z != y, with (x+y+z) % a == 0
        # i.e., z ≡ -(x+y) (mod a)
        # Build: for each residue r, the set of values in others with that residue
        residue_map = {}
        for x in others:
            r = x % a
            if r not in residue_map:
                residue_map[r] = set()
            residue_map[r].add(x)
        
        found = False
        for i in range(n):
            x = others[i]
            for j in range(i+1, n):
                y = others[j]
                needed = (-(x + y)) % a
                if needed in residue_map:
                    candidates = residue_map[needed]
                    # Need z distinct from x and y
                    for z in candidates:
                        if z != x and z != y:
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if found:
            return False
    return True

def _greedy_construct(order, limit=48):
    """Greedy: add elements in given order if valid."""
    selected = []
    for x in order:
        if len(selected) >= limit:
            break
        candidate = selected + [x]
        if _is_valid(candidate):
            selected.append(x)
    return selected

def _local_improve(selected, rng, deadline):
    """Try to improve the set by adding, removing, or swapping elements."""
    selected = list(selected)
    selected_set = set(selected)
    n = 192
    all_vals = set(range(1, n+1))
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding
        for x in sorted(all_vals - selected_set, key=lambda v: v, reverse=True):
            if time.time() >= deadline:
                break
            if len(selected) >= 48:
                break
            candidate = selected + [x]
            if _is_valid(candidate):
                selected = candidate
                selected_set.add(x)
                improved = True
                break
        if improved:
            continue
        # Try swapping: remove one, add one or two
        # Try remove one element, then see if we can add more
        for i in range(len(selected)):
            if time.time() >= deadline:
                break
            without = selected[:i] + selected[i+1:]
            # Try to add elements to without
            without_set = set(without)
            added = False
            for x in sorted(all_vals - without_set, key=lambda v: v, reverse=True):
                if time.time() >= deadline:
                    break
                if len(without) >= 48:
                    break
                candidate = without + [x]
                if _is_valid(candidate):
                    # Check if this gives us more than before
                    if len(without) + 1 > len(selected):
                        selected = without + [x]
                        selected_set = set(selected)
                        improved = True
                        added = True
                        break
            if added:
                break
        if improved:
            continue
        # Try random swaps
        for _ in range(100):
            if time.time() >= deadline:
                break
            if len(selected) < 2:
                break
            # Pick a random element to remove
            i = rng.randrange(len(selected))
            without = selected[:i] + selected[i+1:]
            without_set = set(without)
            # Try to add 1 or 2 random elements
            available = list(all_vals - without_set)
            if not available:
                continue
            # Try adding one
            x = rng.choice(available)
            candidate = without + [x]
            if _is_valid(candidate) and len(candidate) > len(selected):
                selected = candidate
                selected_set = set(selected)
                improved = True
                break
            # Try adding two
            if len(available) >= 2:
                x1, x2 = rng.sample(available, 2)
                candidate = without + [x1, x2]
                if _is_valid(candidate) and len(candidate) > len(selected):
                    selected = candidate
                    selected_set = set(selected)
                    improved = True
                    break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s
    
    rng = random.Random(42)
    
    best = []
    
    # Phase 1: Multiple greedy constructions with different orderings
    orderings = [
        list(range(192, 0, -1)),
        list(range(1, 193)),
        list(range(96, 0, -1)) + list(range(192, 96, -1)),
    ]
    
    # Add some random orderings
    for _ in range(20):
        shuffled = list(range(1, 193))
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    for order in orderings:
        if time.time() >= deadline:
            break
        result = _greedy_construct(order)
        if len(result) > len(best):
            best = result
    
    # Phase 2: Local improvement on the best
    if time.time() < deadline:
        best = _local_improve(best, rng, deadline)
    
    # Phase 3: More random restarts if time permits
    if time.time() < deadline - 5:
        for _ in range(50):
            if time.time() >= deadline:
                break
            shuffled = list(range(1, 193))
            rng.shuffle(shuffled)
            result = _greedy_construct(shuffled)
            if len(result) > len(best):
                best = result
            # Quick local improve
            if len(result) >= len(best) and time.time() < deadline - 2:
                improved = _local_improve(result, rng, deadline - 1)
                if len(improved) > len(best):
                    best = improved
    
    # Final: try to squeeze more in
    if time.time() < deadline:
        best = _local_improve(best, rng, deadline)
    
    return best
# EVOLVE-BLOCK-END
