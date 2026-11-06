"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The set {66, 67, ..., 100} has no subset summing to 200.
    # Min sum of 3 distinct elements: 66 + 67 + 68 = 201 > 200.
    # Max sum of 2 distinct elements: 99 + 100 = 199 < 200.
    # Thus, no subset of size 2 or 3 can sum to 200.
    # Subsets of size >= 4 have min sum 66+67+68+69 = 230 > 200.
    # So the set is valid.
    # Cardinality: 100 - 66 + 1 = 35.
    return list(range(66, 101))
# EVOLVE-BLOCK-END
