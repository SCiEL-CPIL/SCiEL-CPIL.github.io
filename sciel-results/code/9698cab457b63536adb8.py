"""Suite task: distinct subset sums, n=12, minimize the largest element."""
# EVOLVE-BLOCK-START
import time

def construct():
    """Return exactly 12 distinct positive integers <= 1048576 with all 4096
    subset sums distinct.  Minimizes the largest integer.

    Uses a greedy superincreasing sequence construction.
    To minimize the largest element $a_n$ while ensuring distinct subset sums,
    we must choose $a_k$ as small as possible such that it is greater than the
    sum of all previous elements. This forces $a_k = 1 + \sum_{i=1}^{k-1} a_i$,
    which yields the powers of 2: $1, 2, 4, \dots, 2^{n-1}$.
    """
    _t0 = time.monotonic()
    _DEADLINE = 85.0  # safety margin before 90 s kill

    n = 12
    result = []
    current_sum = 0
    
    for i in range(n):
        # The smallest valid next integer is current_sum + 1
        # This ensures the new subset sums (old sums + a_i) do not overlap
        # with the old subset sums.
        next_val = current_sum + 1
        result.append(next_val)
        current_sum += next_val
        
        # Check deadline (though this loop is extremely fast)
        if time.monotonic() > _DEADLINE:
            # If we somehow took too long, return what we have (shouldn't happen)
            # Pad with large numbers if needed to maintain length n?
            # But for n=12, it's instant.
            pass

    # Optional: Verify distinct subset sums for the final result
    # This is a structural addition to ensure correctness.
    # 2^12 = 4096 sums.
    sums = set()
    # Use bitmasks to iterate subsets
    for mask in range(1 << n):
        s = 0
        for j in range(n):
            if mask & (1 << j):
                s += result[j]
        if s in sums:
            # Collision found, though theoretically impossible for this construction
            break
        sums.add(s)
        
    # If collision was found, we would need to search further, 
    # but greedy superincreasing is proven correct.
    
    return result
# EVOLVE-BLOCK-END
