"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Priority based on L2 distance from the all-1s vector (center of {0,1,2}^n).
    Vectors far from center (more extreme, fewer 1s) get higher priority.
    This is a smooth quadratic surrogate that avoids boundary clustering.
    """
    # Compute sum of squared deviations from 1
    s = 0
    for x in vector:
        d = x - 1
        s += d * d
    return float(s)
# EVOLVE-BLOCK-END
