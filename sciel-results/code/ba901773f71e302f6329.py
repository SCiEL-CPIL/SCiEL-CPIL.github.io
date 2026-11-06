"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Prioritizes non-zero vectors with lower Hamming weight.
    The zero vector is deprioritized to avoid early conflicts that restrict
    the inclusion of pairs of vectors {u, -u}. By adding non-zero vectors first,
    we allow a denser packing of the space before considering the zero vector.
    """
    # Calculate Hamming weight (number of non-zero elements)
    weight = 0
    is_zero = True
    for v in vector:
        if v != 0:
            weight += 1
            is_zero = False
    
    if is_zero:
        return -1000.0
    return -float(weight)
# EVOLVE-BLOCK-END
