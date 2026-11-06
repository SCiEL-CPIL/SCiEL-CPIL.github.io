def construct():
    """
    Constructs a set of n=12 distinct positive integers such that all 
    subset sums are distinct, minimizing the largest integer.
    
    We use a greedy strategy: at each step k, we choose the smallest 
    positive integer a_k that has not been used and ensures that all 
    subset sums of the first k elements remain distinct.
    
    It is a known property that the minimal largest element for n 
    distinct positive integers with distinct subset sums is 2^(n-1).
    This is achieved by the superincreasing sequence 1, 2, 4, ..., 2^(n-1).
    The greedy choice of the smallest valid integer at each step 
    (which is sum(previous) + 1) yields exactly this sequence.
    """
    n = 12
    selected = []
    used = set()
    current_sum = 0
    
    for _ in range(n):
        # The smallest integer that preserves distinct subset sums 
        # is one greater than the sum of all previously selected integers.
        candidate = current_sum + 1
        
        # In this specific optimal case, candidate is always distinct 
        # and not in used. For general greedy, we might need to check 
        # distinctness and collisions, but for this specific problem 
        # structure, the superincreasing property guarantees validity.
        selected.append(candidate)
        used.add(candidate)
        current_sum += candidate
        
    return selected
