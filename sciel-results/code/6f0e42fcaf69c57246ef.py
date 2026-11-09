"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Use powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256
    # All subset sums are distinct (binary representation), so sums of
    # different cardinalities are automatically disjoint.
    # All values are in 1..256 and distinct.
    result = []
    val = 1
    while val <= 256:
        result.append(val)
        val *= 2
    # Ensure we return Python ints in a list
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
