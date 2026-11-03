    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Uses a rank-26 decomposition based on Strassen's algorithm for the 2x2 core
    and standard multiplication for the remaining border terms.
    This is the best known constructive rank.
    """
