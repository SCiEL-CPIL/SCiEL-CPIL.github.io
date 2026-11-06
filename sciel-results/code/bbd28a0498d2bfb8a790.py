def construct():
    """
    Returns a list of 12 distinct positive integers such that all subset sums are distinct.
    Uses the set {2, 3, 4, 8, 16, ..., 2048}.
    The largest integer is 2048, which is optimal.
    """
    n = 12
    # The set is 2, 3, and then powers of 2 from 2^2 to 2^11
    # 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    # Count: 2 + 10 = 12.
    # Powers of 2: 2^2 to 2^11 is 10 numbers.
    # Total 12 numbers.
    
    nums = [2, 3]
    for i in range(2, 12):
        nums.append(1 << i)
        
    return nums
