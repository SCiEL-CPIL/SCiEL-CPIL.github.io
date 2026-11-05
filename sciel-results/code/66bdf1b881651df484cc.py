"""Rewritten program: find a 12-element Sidon-like set with distinct subset sums, minimizing max element."""
import time
import random
from itertools import combinations

def _all_subset_sums_distinct(arr):
    """Check that all 2^n subset sums are distinct."""
    n = len(arr)
    sums = set()
    # Use iterative method to generate all subset sums
    # For n=12, 4096 sums is manageable
    current = [0]
    for x in arr:
        current = current + [s + x for s in current]
    if len(current) != len(set(current)):
        return False
    return True

def _max_element(arr):
    return max(arr)

def _construct_greedy():
    """Greedy construction: pick smallest possible next element that keeps all subset sums distinct."""
    arr = []
    current_sums = {0}
    for _ in range(12):
        # Try candidate values starting from len(arr)+1 (since elements must be positive and distinct)
        # Actually, elements just need to be positive integers, distinct, and subset sums unique
        # Start from 1 and go up
        candidate = 1
        while True:
            new_sums = current_sums | {s + candidate for s in current_sums}
            if len(new_sums) == len(current_sums) * 2:
                # All new sums are distinct and don't collide with existing
                arr.append(candidate)
                current_sums = new_sums
                break
            candidate += 1
    return arr

def _construct_local_search():
    """Start from greedy solution and try local improvements to reduce max element."""
    arr = _construct_greedy()
    current_best = arr[:]
    current_max = max(arr)
    
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline with 5s safety margin
    
    # Try to reduce the largest element by swapping or replacing
    # Strategy: try to replace the largest element with a smaller one, or rearrange
    best = arr[:]
    best_max = max(arr)
    
    # Local search: try swapping pairs, or replacing elements with smaller candidates
    # Since we need distinct subset sums, we need to verify after each change
    # For n=12, verification is 4096 sums which is fast
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Pick a random element to try to reduce
        idx = random.randint(0, 11)
        old_val = arr[idx]
        
        # Try replacing with a smaller value
        if old_val <= 1:
            continue
        
        # Try values from 1 to old_val - 1, but skip values already in arr
        existing = set(arr)
        existing.discard(old_val)
        
        # Try a few random smaller values
        candidates = []
        for v in range(1, old_val):
            if v not in existing:
                candidates.append(v)
        
        if not candidates:
            continue
            
        # Try a random subset of candidates
        random.shuffle(candidates)
        for new_val in candidates[:20]:
            trial = arr[:]
            trial[idx] = new_val
            # Sort to maintain consistency (not strictly needed for subset sum check)
            # Check if all subset sums are distinct
            if _all_subset_sums_distinct(trial):
                if max(trial) < best_max:
                    best_max = max(trial)
                    best = trial[:]
                arr = trial[:]
                break
    
    return best

def construct():
    """Construct 12 distinct positive integers <= 1048576 with all subset sums distinct, minimizing max."""
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Start with greedy
    arr = _construct_greedy()
    best = arr[:]
    best_max = max(arr)
    
    # Verify the greedy solution is valid
    assert _all_subset_sums_distinct(best), "Greedy solution invalid"
    assert len(best) == 12
    assert all(x > 0 for x in best)
    assert len(set(best)) == 12
    assert max(best) <= 1048576
    
    # Try local search to improve
    if time.time() < deadline:
        improved = _construct_local_search()
        if max(improved) < best_max:
            best = improved
            best_max = max(improved)
    
    # Final verification
    if not _all_subset_sums_distinct(best):
        # Fallback to greedy
        best = _construct_greedy()
    
    return best
