"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Prioritizes vectors using a deterministic pseudo-random hash.
    This approach breaks the symmetry inherent in weight-based heuristics,
    preventing the greedy algorithm from clustering vectors in a specific
    region (like the origin) which can lead to early saturation of
    arithmetic progressions. A randomized order often allows for a more
    uniform distribution of selected vectors across the space.
    """
    h = 0
    for v in vector:
        h = (h * 31 + v) % 1000000007
    return float(h)
# EVOLVE-BLOCK-END
