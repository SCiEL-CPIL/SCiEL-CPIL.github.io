"""Maximal Sidon set in [1,256]: minimize cardinality via greedy set-cover."""
import time
import random
import sys

N = 256
ALL = set(range(1, N + 1))

def _is_sidon(selected):
    """Check if selected is a Sidon set (all unordered sums a+b, a<=b, distinct)."""
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            b = selected[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _blocked_by(selected, sums):
    """Return set of elements in [1,N] that are blocked by the Sidon set.
    An element y is blocked if adding it would create a duplicate sum.
    y is blocked if for some a in selected: y+a is in sums (and y != a, or 2y already in sums),
    or 2y is in sums."""
    blocked = set()
    for y in ALL:
        # Check if adding y creates a conflict
        # New sums involving y: y+y=2y, and y+a for a in selected (a != y)
        # Conflict if any of these new sums is already in sums, or if two new sums collide
        # 2y conflicts if 2y in sums
        # y+a conflicts if y+a in sums (for a in selected, a != y)
        # Also y+a1 == y+a2 impossible since a1!=a2
        # 2y == y+a means y==a, so if y in selected it's already handled
        is_blocked = False
        if 2 * y in sums:
            is_blocked = True
        else:
            for a in selected:
                if a == y:
                    continue
                if y + a in sums:
                    is_blocked = True
                    break
        if is_blocked:
            blocked.add(y)
    return blocked

def _new_sums_for(selected, y, sums):
    """Return the set of new sums that would be added if y is added to selected."""
    new_sums = set()
    new_sums.add(2 * y)
    for a in selected:
        if a != y:
            new_sums.add(y + a)
    return new_sums

def _can_add(selected, sums, y):
    """Check if y can be added to selected while maintaining Sidon property."""
    # y must not already be in selected
    if y in selected:
        return False
    # New sums: 2y and y+a for a in selected
    # Conflict if 2y in sums, or y+a in sums for any a in selected
    # Also need to check that new sums don't collide with each other (they don't since all involve y)
    if 2 * y in sums:
        return False
    for a in selected:
        if y + a in sums:
            return False
    return True

def _greedy_maximize_blocking(selected, sums, deadline):
    """Iteratively add the element that blocks the most currently-unblocked elements."""
    while time.time() < deadline:
        blocked = _blocked_by(selected, sums)
        unblocked = ALL - blocked
        if not unblocked:
            break  # All elements are blocked, set is maximal
        
        # For each candidate y in unblocked, check if it can be added, and how many new elements it blocks
        best_y = None
        best_new_blocked = 0
        best_new_sums = None
        
        for y in unblocked:
            if not _can_add(selected, sums, y):
                continue
            # Simulate adding y
            new_sums = sums | _new_sums_for(selected, y, sums)
            new_selected = selected + [y]
            new_blocked = _blocked_by(new_selected, new_sums)
            new_blocked_count = len(new_blocked - blocked)
            if new_blocked_count > best_new_blocked:
                best_new_blocked = new_blocked_count
                best_y = y
                best_new_sums = new_sums
        
        if best_y is None:
            break  # No element can be added
        
        # Add the best element
        selected = selected + [best_y]
        sums = best_new_sums
    
    return selected, sums

def _try_random_seed(seed_size, deadline, rng):
    """Try to build a maximal Sidon set starting from a random seed."""
    # Pick a random seed
    candidates = list(ALL)
    rng.shuffle(candidates)
    
    # Try to build a Sidon set of size seed_size
    selected = []
    sums = set()
    for y in candidates:
        if len(selected) >= seed_size:
            break
        if _can_add(selected, sums, y):
            selected.append(y)
            new_sums = _new_sums_for(selected[:-1], y, sums)
            sums = sums | new_sums
    
    if len(selected) < seed_size:
        return None
    
    # Now greedily add elements to maximize blocking
    selected, sums = _greedy_maximize_blocking(selected, sums, deadline)
    
    # Verify it's maximal
    blocked = _blocked_by(selected, sums)
    if blocked == ALL:
        return selected
    return None

def construct():
    """Construct a minimal-size inclusion-maximal Sidon set in [1,256]."""
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Try different seed sizes, starting small
    best = None
    best_size = N + 1
    
    # Deterministic seed for reproducibility
    rng = random.Random(42)
    
    # Try seed sizes from 5 upward
    for seed_size in range(5, 15):
        if time.time() > deadline:
            break
        
        # Try multiple random seeds for this size
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            result = _try_random_seed(seed_size, deadline, rng)
            if result is not None:
                if len(result) < best_size:
                    best = result
                    best_size = len(result)
                # If we found a small one, we can stop trying larger seeds
                if best_size <= seed_size + 3:
                    break
        
        # If we found a good solution, stop
        if best_size <= seed_size + 2:
            break
    
    # If we didn't find anything, fall back to the original greedy
    if best is None:
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected
    
    return best
