"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Returns a list of 31 distinct integers in 1..256
    # satisfying the cardinality-separated subset sum property.
    # The set {226, ..., 256} is known to be optimal.
    start_val = 226
    end_val = 257
    result = []
    for i in range(start_val, end_val):
        result.append(i)
    return result
# EVOLVE-BLOCK-END
