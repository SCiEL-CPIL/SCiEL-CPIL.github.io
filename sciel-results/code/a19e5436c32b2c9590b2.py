"""Search-based approach to maximize set size under divisibility constraint."""
import time
import random

# EVOLVE-BLOCK-START

DEADLINE_SECONDS = 85.0

def _all_subset_sums_fast(elements):
    """Generate all nonempty subset sums using iterative DP."""
    sums = set()
    for e in elements:
        # Create a list of current sums to iterate over safely while updating set
        # This avoids "set changed size during iteration" issues if we were doing it differently,
        # but here we build a new set of additions.
        current_sums = list(sums)
        for s in current_sums:
            sums.add(s + e)
        sums.add(e)
    return sums

def _check_divisibility(sums):
    """Check if no two distinct subset sums divide one another."""
    if not sums:
        return True
    
    sum_list = sorted(sums)
    sum_set = sums
    max_sum = sum_list[-1]
    
    for a in sum_list:
        if a == 0:
            continue
        # Check if any multiple of a (2a, 3a, ...) is in the set
        multiple = 2 * a
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += a
    return True

def _is_valid(elements):
    """Check if no two distinct subset sums divide one another."""
    if not elements:
        return True
    sums = _all_subset_sums_fast(elements)
    return _check_divisibility(sums)

def _is_valid_incremental(elements, candidate, current_sums):
    """
    Check if adding candidate to elements (with current_sums) is valid.
    Returns (is_valid, new_sums) if valid, else (False, None).
    """
    # Generate new sums
    new_sums = set(current_sums)
    # Add sums that include the candidate
    for s in current_sums:
        new_sums.add(s + candidate)
    new_sums.add(candidate)
    
    if not _check_divisibility(new_sums):
        return False, None
    
    return True, new_sums

def _local_search(elements, candidates, deadline, max_iter=1000):
    """
    Local search using hill climbing with swaps.
    """
    current = list(elements)
    current_set = set(current)
    current_sums = _all_subset_sums_fast(current)
    
    best = list(current)
    best_sums = set(current_sums)
    best_size = len(current)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        # 1. Try to add a random candidate
        if candidates:
            c = random.choice(candidates)
            if c not in current_set:
                valid, new_sums = _is_valid_incremental(current, c, current_sums)
                if valid:
                    current.append(c)
                    current_set.add(c)
                    current_sums = new_sums
                    if len(current) > best_size:
                        best = list(current)
                        best_sums = set(current_sums)
                        best_size = len(current)
        
        # 2. Try a swap: Remove one, add one
        # This helps escape local maxima where no single add works.
        if len(current) > 1 and candidates:
            remove_idx = random.randint(0, len(current) - 1)
            removed = current[remove_idx]
            
            # Remove from current
            current.pop(remove_idx)
            current_set.discard(removed)
            
            # Recalculate sums for the reduced set (expensive but necessary for correctness of incremental add)
            # We could try to remove 'removed' from sums, but that's complex.
            temp_sums = _all_subset_sums_fast(current)
            
            added = False
            # Try a few random candidates to add
            for _ in range(5):
                c = random.choice(candidates)
                if c not in current_set and c != removed:
                    valid, new_sums = _is_valid_incremental(current, c, temp_sums)
                    if valid:
                        current.append(c)
                        current_set.add(c)
                        current_sums = new_sums
                        added = True
                        break
            
            if not added:
                # Revert
                current.append(removed)
                current_set.add(removed)
                current_sums = _all_subset_sums_fast(current)

    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    all_candidates = list(range(1, 257))
    random.seed(42)
    
    best_result = []
    best_size = 0
    
    # Multiple restarts
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Start with a small random set of larger numbers
        # Larger numbers tend to have "larger" sums, potentially reducing divisibility conflicts
        # with smaller sums? Not necessarily, but it's a heuristic.
        start_size = random.randint(3, 5)
        start_elements = random.sample([x for x in all_candidates if x >= 50], start_size)
        
        if not _is_valid(start_elements):
            continue
        
        current = list(start_elements)
        current_sums = _all_subset_sums_fast(current)
        
        # Greedy expansion
        candidates = [x for x in all_candidates if x not in set(current)]
        random.shuffle(candidates)
        
        for c in candidates:
            if time.time() > deadline:
                break
            if c in set(current):
                continue
            valid, new_sums = _is_valid_incremental(current, c, current_sums)
            if valid:
                current.append(c)
                current_sums = new_sums
        
        # Local search
        current = _local_search(current, candidates, deadline, max_iter=200)
        
        if len(current) > best_size:
            best_result = list(current)
            best_size = len(current)
            
        if best_size >= 15:
            break

    if best_result and not _is_valid(best_result):
        best_result = [31, 32, 34]
        
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
