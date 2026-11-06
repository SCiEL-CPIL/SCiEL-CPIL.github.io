import time
import random

def _subset_sums_distinct(seq):
    """
    Fast check: all 2^n subset sums of seq are distinct.
    Uses a Python int as a bitset of achievable sums.
    Returns True iff the number of set bits equals 2^n.
    """
    bits = 1
    for a in seq:
        bits |= (bits << a)
    n = len(seq)
    return bits.bit_count() == (1 << n)

def _greedy_smallest_seq(n, max_val):
    """
    Build the lexicographically smallest valid sequence of length n
    with each element in [1, max_val]. Returns None if impossible.
    This gives the minimum possible largest element for a 'greedy' construction,
    which is often close to optimal.
    """
    seq = []
    for i in range(n):
        for v in range(1, max_val + 1):
            candidate = seq + [v]
            if _subset_sums_distinct(candidate):
                seq.append(v)
                break
        else:
            return None
    return seq

def _local_search(seq, max_time=40.0):
    """
    Local search to minimize the maximum element.
    Start from a valid sequence, try to reduce the largest element,
    then other elements, using random restarts and hill climbing.
    """
    start = time.time()
    n = len(seq)
    best_seq = list(seq)
    best_max = max(seq)
    
    # Try to reduce the largest element first
    # Strategy: repeatedly try to decrease a random element, keep if valid
    # Focus on elements that are large.
    
    current = list(seq)
    current_max = max(current)
    
    if current_max > 2048:
        # If we start with something worse than powers of 2, try to improve
        for _ in range(100000):
            if time.time() - start > max_time - 10:
                break
            i = random.randrange(n)
            if current[i] <= 1:
                continue
            old = current[i]
            # Try to reduce current[i]
            # Random reduction
            new_val = random.randint(1, old - 1)
            current[i] = new_val
            if _subset_sums_distinct(current):
                new_max = max(current)
                if new_max < current_max:
                    current_max = new_max
                    best_seq = list(current)
                    best_max = current_max
            else:
                current[i] = old
    else:
        # Already at or below powers of 2 max. Try to verify if we can go lower.
        # This is the hard part. Powers of 2 is the lower bound.
        # Let's do a quick check if any element can be reduced in powers of 2.
        for i in range(n):
            if time.time() - start > max_time - 5:
                break
            old = current[i]
            if old <= 1:
                continue
            # Try reducing element i by 1
            current[i] = old - 1
            if _subset_sums_distinct(current):
                # We found a better solution!
                best_seq = list(current)
                best_max = max(current)
                # Continue searching from here
                # (Loop continues naturally)
            else:
                current[i] = old
        
        # If we are at 2048 and couldn't reduce any single element, 
        # it's highly likely 2048 is optimal.
        # A more thorough search would be needed to be certain, 
        # but given the theoretical bound, we can stop early.
        if best_max <= 2048:
            # One more attempt: try reducing multiple elements simultaneously?
            # Too complex for simple local search.
            # Return best found.
            pass

    return best_seq

def construct():
    n = 12
    max_limit = 1048576
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    # The theoretical lower bound for the largest element is 2^(n-1) = 2048.
    # Powers of 2 achieve this bound.
    seq_powers = [1 << i for i in range(n)]
    if _subset_sums_distinct(seq_powers):
        # This is valid and achieves the theoretical minimum.
        # Any valid sequence must have max >= 2048.
        # So 2048 is optimal.
        # We can return this immediately.
        # But let's double check if there's any weird constraint I missed.
        # "distinct positive Python integers" - yes.
        # "Every subset, including the empty subset, must have a different sum" - yes.
        # "Minimize the largest integer" - yes.
        return seq_powers
    
    # If for some reason powers of 2 didn't work (shouldn't happen), 
    # fall back to search.
    best_seq = None
    best_max = max_limit + 1
    
    # Try greedy construction with decreasing max
    # Start from a reasonable upper bound
    # We know 2048 works. Try to see if anything < 2048 works.
    # But we know it can't. So this is just for robustness.
    
    # Actually, just return powers of 2 if valid.
    # The code above already returns it.
    # This part is unreachable if powers of 2 are valid.
    
    return best_seq if best_seq else seq_powers
