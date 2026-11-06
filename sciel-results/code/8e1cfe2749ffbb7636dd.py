"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 12
    # Greedy construction of a superincreasing sequence.
    # To ensure distinct subset sums, each element must be strictly greater
    # than the sum of all previous elements.
    # To minimize the largest element, we choose the smallest possible integer
    # at each step, which is sum_prev + 1.
    
    seq = []
    current_sum = 0
    
    for i in range(n):
        if i == 0:
            # The first element must be at least 1.
            # Choosing 1 minimizes the sum for subsequent steps.
            next_val = 1
        else:
            # The next element must be > current_sum.
            # Smallest such integer is current_sum + 1.
            next_val = current_sum + 1
        
        seq.append(next_val)
        current_sum += next_val
        
    # The sequence is [1, 2, 4, ..., 2048]
    # Largest element is 2048.
    return seq
# EVOLVE-BLOCK-END
