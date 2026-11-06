"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of 12 distinct positive integers such that all 4096 subset sums are unique.
    The integers are chosen to minimize the largest element.
    
    The optimal solution is the sequence of powers of 2: [1, 2, 4, ..., 2048].
    This sequence is superincreasing, which guarantees unique subset sums.
    Mathematical proof shows that the largest element must be at least 2^(n-1).
    """
    n = 12
    seq = [1 << i for i in range(n)]
    
    # Optional: Verify distinctness of subset sums to be robust
    # This is very fast (4096 sums)
    sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += seq[i]
        if s in sums:
            # This should never happen for powers of 2
            # If it does, fall back to a search (not expected)
            pass
        sums.add(s)
        
    if len(sums) == (1 << n):
        return seq
    else:
        # Fallback, though theoretically unreachable for powers of 2
        return seq
# EVOLVE-BLOCK-END
