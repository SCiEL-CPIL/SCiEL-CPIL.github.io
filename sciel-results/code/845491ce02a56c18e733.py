def construct():
    """
    Constructs a set of 12 distinct positive integers with distinct subset sums
    by greedily selecting the smallest possible next integer at each step.
    This yields the powers of 2, which is the optimal solution for minimizing
    the largest element.
    """
    n = 12
    # Start with the empty subset sum: 0
    sums = {0}
    result = []
    
    for _ in range(n):
        # Find the smallest positive integer x not in result
        # such that adding x creates no duplicate sums
        x = 1
        while True:
            # Check if x is already in the result (distinct elements constraint)
            if x in result:
                x += 1
                continue
            
            # Calculate new sums
            new_sums = sums | {s + x for s in sums}
            
            # If the number of unique sums doubles, there are no collisions
            if len(new_sums) == 2 * len(sums):
                result.append(x)
                sums = new_sums
                break
            else:
                x += 1
                
    return result
