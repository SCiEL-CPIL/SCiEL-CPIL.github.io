"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    # Prioritize vectors with higher sum of coordinates.
    # This deprioritizes the zero vector and low-weight vectors,
    # which are more likely to form 3-term arithmetic progressions
    # with each other (e.g., 0, e_i, 2e_i).
    # Vectors with higher sums are 'more complex' and less likely to
    # form simple additive dependencies early in the greedy selection.
    return float(sum(vector))
# EVOLVE-BLOCK-END
