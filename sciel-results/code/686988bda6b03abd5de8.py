import time
import itertools
from collections import deque

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

def _subset_sums_set(seq):
    """Return the set of all subset sums for seq."""
    sums = {0}
    for x in seq:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        sums |= new_sums
    return sums

def _check_validity(seq):
    """Check if seq has distinct subset sums."""
    n = len(seq)
    return len(_subset_sums_set(seq)) == (1 << n)

def _greedy_construct(n):
    """
    Greedy construction: build sequence element by element.
    At each step, choose the smallest positive integer not already used
    that maintains distinct subset sums.
    """
    seq = []
    used = set()
    current_sums = {0}
    
    for i in range(n):
        # Try candidates starting from 1
        candidate = 1
        while True:
            if candidate not in used:
                # Check if adding candidate maintains distinct subset sums
                new_sums = set(current_sums)
                for s in current_sums:
                    new_sums.add(s + candidate)
                if len(new_sums) == len(current_sums) + len(current_sums):
                    # All subset sums remain distinct
                    seq.append(candidate)
                    used.add(candidate)
                    current_sums = new_sums
                    break
            candidate += 1
    
    return seq

def _local_search_improve(seq, deadline):
    """
    Try to improve the sequence by reducing elements.
    Use a local search approach: try to reduce each element to a smaller value.
    """
    n = len(seq)
    best = list(seq)
    best_max = max(best)
    
    # Sort indices by value (largest first) to prioritize reducing large elements
    indices = sorted(range(n), key=lambda i: -best[i])
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in indices:
            current_val = best[idx]
            if current_val <= 1:
                continue
            # Try to reduce this element
            # The new value must be positive, distinct from others, and maintain subset sum distinctness
            others = [best[j] for j in range(n) if j != idx]
            others_set = set(others)
            
            # Try values from current_val-1 down to 1
            # But we can be smarter: the minimum valid value is constrained
            # For efficiency, try a limited range first
            for new_val in range(current_val - 1, 0, -1):
                if new_val in others_set:
                    continue
                # Check validity
                candidate = list(best)
                candidate[idx] = new_val
                if _check_validity(candidate):
                    best = candidate
                    best_max = max(best)
                    improved = True
                    break
    
    return best, best_max

def _try_superincreasing_with_gaps(n, deadline):
    """
    Try to find a superincreasing-like sequence but with gaps to minimize the max.
    A superincreasing sequence requires a[i] > sum(a[0..i-1]), which gives powers of 2.
    But we can try non-superincreasing sequences.
    """
    # Use greedy construction which already tries to minimize each element
    seq = _greedy_construct(n)
    return seq

def construct():
    """
    Constructs a set of 12 distinct positive integers with distinct subset sums,
    minimizing the largest element.
    
    Uses greedy construction followed by local search improvement.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    n = 12
    
    # Step 1: Greedy construction
    seq = _greedy_construct(n)
    best_max = max(seq)
    
    # Verify
    if not _check_validity(seq):
        # Fallback to powers of 2
        seq = [1 << j for j in range(n)]
        best_max = seq[-1]
    
    # Step 2: Local search to improve
    if time.time() < deadline:
        seq, best_max = _local_search_improve(seq, deadline)
    
    # Step 3: Try additional improvement strategies
    if time.time() < deadline:
        # Try swapping elements to see if we can reduce the max further
        # Focus on the largest element
        max_idx = seq.index(max(seq))
        current_max = seq[max_idx]
        
        # Try to find a smaller value for the max element
        others = [seq[j] for j in range(n) if j != max_idx]
        others_set = set(others)
        
        # The max element must be large enough to not create collisions
        # Sum of all other elements
        sum_others = sum(others)
        
        # Try values from current_max-1 down
        # But we know the max must be at least sum_others + 1 for superincreasing
        # For non-superincreasing, it could be smaller
        # Try a limited search
        for trial in range(current_max - 1, max(0, sum_others - 100), -1):
            if time.time() > deadline:
                break
            if trial in others_set:
                continue
            candidate = list(seq)
            candidate[max_idx] = trial
            if _check_validity(candidate):
                seq = candidate
                best_max = trial
                break
    
    return seq
