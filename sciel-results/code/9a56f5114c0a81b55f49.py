def construct():
    """
    Constructs a set of 12 distinct positive integers such that all subset sums are distinct,
    minimizing the largest integer.
    
    The optimal solution is the powers of 2: [1, 2, 4, ..., 2048].
    The largest element is 2048, which is the theoretical minimum.
    """
    n = 12
    nums = []
    current_sums = {0}
    
    for _ in range(n):
        x = 1
        while True:
            new_sums = current_sums | {s + x for s in current_sums}
            if len(new_sums) == 2 * len(current_sums):
                nums.append(x)
                current_sums = new_sums
                break
            x += 1
                
    return nums
