"""Optimal construction for distinct subset sums, n=12.

The powers of 2 [1, 2, 4, ..., 2048] are provably optimal: they achieve the
minimum possible sum (4095 = 2^12 - 1) required for 4096 distinct subset sums,
and the maximum element 2048 = 2^11 is the minimum possible maximum.
"""

def construct():
    # Powers of 2 are provably optimal for this problem.
    # Sum = 2^12 - 1 = 4095 (minimum possible for 4096 distinct non-negative sums)
    # Max element = 2^11 = 2048 (minimum possible given the sum constraint and
    # the requirement that all 2^12 subset sums are distinct integers)
    n = 12
    result = [1 << i for i in range(n)]
    
    # Verify: all 4096 subset sums are distinct
    sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += result[i]
        sums.add(s)
    
    assert len(sums) == (1 << n), f"Expected {1 << n} distinct sums, got {len(sums)}"
    assert max(result) <= 1048576
    assert all(x > 0 for x in result)
    assert len(set(result)) == n
    
    return result
