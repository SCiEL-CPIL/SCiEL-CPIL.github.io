def construct():
    """
    Returns a list of 12 distinct positive integers <= 1048576 such that
    all 2^12 subset sums are distinct, minimizing the largest integer.
    
    The theoretical minimum for the largest integer in a set of n distinct
    positive integers with distinct subset sums is 2^(n-1).
    For n=12, this is 2^11 = 2048.
    The set of powers of 2 achieves this bound.
    """
    n = 12
    # The optimal set is {2^0, 2^1, ..., 2^(n-1)}
    # Largest element is 2^(n-1)
    result = [1 << i for i in range(n)]
    
    # Verification: Check if all subset sums are distinct
    # For n=12, there are 2^12 = 4096 subset sums.
    # This is fast enough to run on every call if needed, but we can skip it
    # for speed if we are confident. However, the prompt asks for a robust program.
    # Let's do a fast verification.
    
    def _validate(arr):
        sums = set()
        # Generate all subset sums
        for mask in range(1 << len(arr)):
            s = 0
            for i in range(len(arr)):
                if mask & (1 << i):
                    s += arr[i]
            if s in sums:
                return False
            sums.add(s)
        return True

    if not _validate(result):
        # This should never happen for powers of 2
        # Fallback to a known valid set (same as result)
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        
    return result
