"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256
DEADLINE_MARGIN = 0.5  # seconds before the 30s limit to stop searching


# EVOLVE-BLOCK-START
def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set (all unordered sums a+b with a<=b distinct)."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _can_add(selected, sums, x):
    """Check if x can be added to selected without breaking Sidon property."""
    new_sums = set()
    for a in selected:
        s = a + x
        if s in sums or s in new_sums:
            return False
        new_sums.add(s)
    s2 = 2 * x
    if s2 in sums or s2 in new_sums:
        return False
    return True


def _build_sums(selected):
    """Build the set of all unordered sums for a Sidon set."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    return sums


def _is_maximal(selected, sums):
    """Check if the Sidon set is inclusion-maximal in [1,N]."""
    for x in range(1, N + 1):
        if x not in selected and _can_add(selected, sums, x):
            return False
    return True


def _greedy_randomized(seed=None, deadline=None):
    """
    Greedy randomized construction: pick elements in random order,
    add if it doesn't break Sidon property. Then try to maximize by
    checking if it's maximal; if not, we just return what we have
    (the problem asks for maximal, so we need to ensure maximality).
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Random permutation of [1, N]
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    selected = []
    sums = set()
    
    for x in candidates:
        if _can_add(selected, sums, x):
            selected.append(x)
            # Update sums
            for a in selected[:-1]:  # all previously selected
                sums.add(a + x)
            sums.add(2 * x)
        
        if deadline is not None and time.time() > deadline:
            break
    
    # Check maximality; if not maximal, try to force maximality
    # by adding more elements (but we can't add if it breaks Sidon)
    # Actually, if it's not maximal, there exists some x we could add.
    # But our greedy already tried all candidates in random order.
    # If we couldn't add x, it's because it conflicts. So if we went
    # through all candidates, the set IS maximal (no element can be added).
    
    return selected


def _greedy_with_backtrack_heuristic(seed=None, deadline=None):
    """
    A smarter greedy: prioritize elements that block the most other elements.
    This tends to produce smaller maximal sets.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    selected = []
    sums = set()
    remaining = set(range(1, N + 1))
    
    while remaining:
        if deadline is not None and time.time() > deadline:
            break
        
        # Score each remaining element by how many other elements it would block
        # (i.e., how many elements y in remaining would conflict with x)
        best_x = None
        best_score = -1
        best_conflicts = set()
        
        # Sample a subset for efficiency if remaining is large
        sample = list(remaining)
        if len(sample) > 50:
            sample = rng.sample(sample, 50)
        
        for x in sample:
            # Count how many elements in remaining would conflict with x
            conflicts = 0
            for y in remaining:
                if y == x:
                    continue
                # Check if x and y can coexist
                # x+y must not be in sums, and 2x, 2y must not be in sums
                # Also x+y must not equal 2z for any z in selected, etc.
                # Simplified: check if adding x to selected is valid, then check y
                # This is expensive, so use a proxy: count sums that would be created
                s_xy = x + y
                if s_xy in sums:
                    conflicts += 1
                    continue
                # Check if 2x in sums
                if 2 * x in sums:
                    break  # x can't even be added
                # Check if 2y in sums
                if 2 * y in sums:
                    conflicts += 1
                    continue
                # Check if x+y conflicts with existing sums
                # Also need to check all a+x and a+y for a in selected
                ok = True
                for a in selected:
                    if (a + x) in sums or (a + y) in sums:
                        ok = False
                        break
                if not ok:
                    conflicts += 1
                    continue
                # Also check a+b for a,b in selected doesn't equal x+y
                # (already checked s_xy in sums)
                conflicts += 0  # they can coexist
                
            # Higher conflicts means x blocks more, so prefer it
            if conflicts > best_score:
                best_score = conflicts
                best_x = x
        
        if best_x is None:
            # No element can be added? Shouldn't happen if remaining is non-empty
            break
        
        # Add best_x
        if _can_add(selected, sums, best_x):
            selected.append(best_x)
            for a in selected[:-1]:
                sums.add(a + best_x)
            sums.add(2 * best_x)
            remaining.discard(best_x)
        else:
            # Shouldn't happen, but just in case
            remaining.discard(best_x)
    
    return selected


def _local_search_improve(selected, sums, deadline=None):
    """
    Try to improve (reduce size) by removing elements and re-optimizing.
    Actually, we want to find a SMALLER maximal set, not a larger one.
    This is tricky. Instead, let's try different random seeds and keep the smallest maximal set.
    """
    return selected


def construct():
    start_time = time.time()
    deadline = start_time + 30 - DEADLINE_MARGIN
    
    best_set = None
    best_size = float('inf')
    
    # Try multiple randomized greedy approaches
    # Approach 1: Simple random greedy
    for seed in range(100):
        if time.time() > deadline:
            break
        s = _greedy_randomized(seed=seed, deadline=deadline)
        if len(s) < best_size:
            # Verify it's valid and maximal
            sums = _build_sums(s)
            if _is_valid_sidon(s) and _is_maximal(s, sums):
                best_set = s
                best_size = len(s)
    
    # Approach 2: Blocking heuristic greedy
    for seed in range(100):
        if time.time() > deadline:
            break
        s = _greedy_with_backtrack_heuristic(seed=seed, deadline=deadline)
        if len(s) < best_size:
            sums = _build_sums(s)
            if _is_valid_sidon(s) and _is_maximal(s, sums):
                best_set = s
                best_size = len(s)
    
    # Approach 3: Ascending greedy (original) as fallback
    if best_set is None:
        selected = []
        sums = set()
        for x in range(1, N + 1):
            if _can_add(selected, sums, x):
                selected.append(x)
                for a in selected[:-1]:
                    sums.add(a + x)
                sums.add(2 * x)
        best_set = selected
    
    # Final verification
    sums = _build_sums(best_set)
    assert _is_valid_sidon(best_set), "Result is not a valid Sidon set"
    assert _is_maximal(best_set, sums), "Result is not maximal"
    
    return best_set
# EVOLVE-BLOCK-END
