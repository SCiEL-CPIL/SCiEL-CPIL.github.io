import time
import itertools

def _has_distinct_subset_sums(seq):
    """Check if all subset sums are distinct."""
    n = len(seq)
    seen = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += seq[i]
        if s in seen:
            return False
        seen.add(s)
    return True

def construct():
    """
    Constructs a set of 12 distinct positive integers with distinct subset sums,
    minimizing the largest element.
    
    Starts with powers of 2 and attempts local search to improve.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    n = 12
    # Start with powers of 2
    seq = [1 << j for j in range(n)]
    best_max = seq[-1]
    
    # Verify initial solution
    if not _has_distinct_subset_sums(seq):
        # Should never happen for powers of 2
        return seq
    
    # Try local search: try to reduce the largest element
    # First, try to see if we can reduce the last element
    # The last element must be > sum of all others for superincreasing,
    # but maybe we can find non-superincreasing solutions
    
    # Try swapping/reducing elements from the end
    # Strategy: try to reduce the largest element by checking if a smaller value works
    
    # For each position from the end, try to find a smaller valid value
    # This is complex, so let's try a simpler approach:
    # Try to reduce the last element by 1, 2, 3... and check validity
    
    # But this could be slow. Let's try a smarter approach.
    
    # Actually, for n=12, the search space is too large for brute force.
    # Let's try a few targeted improvements.
    
    # Try: can we replace the last element with something smaller?
    # The last element in powers of 2 is 2048.
    # Sum of first 11 elements = 2047.
    # For distinct subset sums, we need the last element to not create collisions.
    
    # A sufficient condition: last element > sum of all previous (superincreasing)
    # But it's not necessary. Let's check if 2047 works (it's equal to sum, so likely fails)
    
    # Let's try a few values below 2048
    for trial in range(2047, max(0, 2048 - 100), -1):
        if time.time() > deadline:
            break
        candidate = seq[:11] + [trial]
        # Check distinctness
        if _has_distinct_subset_sums(candidate):
            best_max = trial
            seq = candidate
            break
    
    # Try to improve earlier elements as well
    # This is more complex; for now, return the best found
    
    return seq
