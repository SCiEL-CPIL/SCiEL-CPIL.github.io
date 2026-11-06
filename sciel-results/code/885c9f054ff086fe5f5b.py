"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 12
    # Construct the optimal superincreasing sequence.
    # The minimal largest element for n distinct positive integers with
    # distinct subset sums is 2^(n-1).
    # This is achieved by the sequence 1, 2, 4, ..., 2^(n-1).
    
    result = []
    total = 0
    for i in range(n):
        if i == 0:
            val = 1
        else:
            val = total + 1
        result.append(val)
        total += val
        
    return result
# EVOLVE-BLOCK-END
