"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _is_valid(a):
    """Check if a sorted list of distinct non-negative ints is a Sidon set covering 1..30."""
    if len(a) > 32:
        return False
    if a[0] != 0:
        return False
    if any(x < 0 or x > 10**6 for x in a):
        return False
    # Check distinctness
    if len(set(a)) != len(a):
        return False
    # Check Sidon property: all positive differences distinct
    diffs = set()
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            if d in diffs:
                return False
            diffs.add(d)
    # Check coverage of 1..30
    for d in range(1, 31):
        if d not in diffs:
            return False
    return True

def _diameter(a):
    return max(a)

def _get_diameter(a):
    return max(a)

def _get_differences(a):
    """Return set of positive differences."""
    diffs = set()
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(a[j] - a[i])
    return diffs

def _is_sidon_and_covers(a):
    """Check Sidon property and coverage of 1..30."""
    if len(a) > 32:
        return False
    if len(set(a)) != len(a):
        return False
    diffs = set()
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            if d in diffs:
                return False
            diffs.add(d)
    for d in range(1, 31):
        if d not in diffs:
            return False
    return True

def _generate_initial_set():
    """Generate a valid Sidon set covering 1..30 using greedy construction."""
    a = [0]
    
    def diffs_of(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    
    for d in range(1, 31):
        if d in diffs_of(sorted(a)):
            continue
        # Try to add a single element
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs_of(trial))) == len(diffs_of(trial)):
                a = trial
                found = True
                break
        if found:
            continue
        # Add two elements
        y = 2 * max(a) + 3 * d + 1
        while True:
            trial = sorted(a + [y, y + d])
            if len(set(diffs_of(trial))) == len(diffs_of(trial)):
                a = trial
                break
            y += 1
    return a

def _local_search(a, time_limit):
    """Perform randomized hill-climbing to reduce the diameter."""
    a = list(a)
    start_time = time.time()
    best = a
    best_diam = _diameter(best)
    
    # Precompute some useful data
    def compute_diffs(s):
        diffs = set()
        n = len(s)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(s[j] - s[i])
        return diffs
    
    # Track the set of differences for quick checking
    current_diffs = compute_diffs(a)
    
    def try_move(old_idx, new_val):
        """Try moving element at old_idx to new_val. Returns (new_a, new_diffs) or None."""
        nonlocal current_diffs
        if new_val < 0 or new_val > 10**6:
            return None
        # Check if new_val collides with any other element
        for i, v in enumerate(a):
            if i != old_idx and v == new_val:
                return None
        
        old_val = a[old_idx]
        
        # Compute differences that will be removed (those involving old_idx)
        removed_diffs = set()
        for i in range(len(a)):
            if i == old_idx:
                continue
            d = abs(a[i] - old_val)
            removed_diffs.add(d)
        
        # Compute differences that will be added (those involving new_val)
        added_diffs = set()
        for i in range(len(a)):
            if i == old_idx:
                continue
            d = abs(a[i] - new_val)
            added_diffs.add(d)
        
        # Check Sidon: no added diff should already be in current_diffs (minus removed)
        remaining_diffs = current_diffs - removed_diffs
        for d in added_diffs:
            if d in remaining_diffs:
                return None
        
        # Check coverage: all 1..30 must be in (remaining_diffs | added_diffs)
        new_diffs = remaining_diffs | added_diffs
        for d in range(1, 31):
            if d not in new_diffs:
                return None
        
        # Check size constraint
        if len(a) > 32:
            return None
        
        new_a = list(a)
        new_a[old_idx] = new_val
        new_a.sort()
        
        return (new_a, new_diffs)
    
    # Randomized hill-climbing
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        if no_improve_count >= max_no_improve:
            # Reset with a random restart
            a = list(best)
            current_diffs = compute_diffs(a)
            no_improve_count = 0
            max_no_improve = 500 + random.randint(0, 500)
        
        idx = random.randint(0, len(a) - 1)
        old_val = a[idx]
        
        # Generate candidate new values
        candidates = set()
        
        # Try values near existing elements
        for v in a:
            for delta in range(1, 20):
                candidates.add(v + delta)
                candidates.add(v - delta)
        
        # Try random values
        for _ in range(30):
            candidates.add(random.randint(0, best_diam + 50))
        
        # Try to reduce diameter
        if idx == len(a) - 1:  # Largest element
            for delta in range(1, min(50, best_diam)):
                candidates.add(old_val - delta)
        
        # Remove old value from candidates
        candidates.discard(old_val)
        
        improved = False
        for new_val in sorted(candidates):
            if time.time() - start_time >= time_limit:
                break
            result = try_move(idx, new_val)
            if result is None:
                continue
            new_a, new_diffs = result
            new_diam = _diameter(new_a)
            if new_diam < best_diam:
                best = new_a
                best_diam = new_diam
                a = new_a
                current_diffs = new_diffs
                improved = True
                no_improve_count = 0
                break
        
        if not improved:
            no_improve_count += 1
    
    return best

def construct():
    """Construct a Sidon set containing 0 with all differences 1..30, minimizing the largest element."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Generate initial solution
    a = _generate_initial_set()
    
    if not _is_valid(a):
        raise ValueError("Initial solution is not valid")
    
    best = a
    best_diam = _diameter(best)
    
    # Run local search with remaining time
    remaining = deadline - time.time()
    if remaining > 5:
        result = _local_search(a, remaining - 5)
        if _is_valid(result) and _diameter(result) < best_diam:
            best = result
            best_diam = _diameter(best)
    
    # Final validation
    if not _is_valid(best):
        # Fallback to the greedy solution
        best = _generate_initial_set()
    
    return best

# EVOLVE-BLOCK-END
