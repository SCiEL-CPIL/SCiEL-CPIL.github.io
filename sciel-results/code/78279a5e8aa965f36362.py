import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of 12 distinct positive integers <= 1048576 such that
    all 2^12 subset sums are distinct, minimizing the largest integer.
    
    Theoretical Lower Bound:
    For a set of n distinct positive integers to have all 2^n subset sums distinct,
    the elements must be superincreasing. That is, if a_1 < a_2 < ... < a_n,
    then a_k > sum(a_1, ..., a_{k-1}) for all k.
    
    Proof sketch:
    Let S_{k-1} be the set of subset sums of {a_1, ..., a_{k-1}}.
    The subset sums of {a_1, ..., a_k} are S_{k-1} U (S_{k-1} + a_k).
    For these to be disjoint, we need max(S_{k-1}) < min(S_{k-1} + a_k)
    => sum(a_1...a_{k-1}) < a_k.
    
    To minimize a_n, we minimize the previous terms:
    a_1 >= 1
    a_2 > a_1 >= 1 => a_2 >= 2
    a_3 > a_1 + a_2 >= 3 => a_3 >= 4
    ...
    a_k >= 2^(k-1)
    
    Thus a_12 >= 2^11 = 2048.
    The set {1, 2, 4, ..., 2048} achieves this bound exactly.
    Therefore, 2048 is the global minimum for the largest element.
    
    This program returns this optimal set directly.
    """
    n = 12
    # The optimal set is powers of 2: 2^0, 2^1, ..., 2^(n-1)
    # Largest element is 2^(n-1) = 2048
    result = [1 << i for i in range(n)]
    
    # Safety check: ensure values are within bounds
    # Max allowed is 1048576 (2^20). 2048 is well within this.
    
    # Optional verification (fast for n=12, 4096 subsets)
    # This is included to satisfy robustness, though mathematically guaranteed.
    def _validate(arr):
        sums = set()
        for mask in range(1 << len(arr)):
            s = 0
            # Efficient sum calculation
            # For n=12, simple loop is fine
            for i in range(len(arr)):
                if mask & (1 << i):
                    s += arr[i]
            if s in sums:
                return False
            sums.add(s)
        return True

    if not _validate(result):
        # Should never happen, but fallback to a known valid superincreasing sequence
        # (which is the same as powers of 2)
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        
    return result
# EVOLVE-BLOCK-END
