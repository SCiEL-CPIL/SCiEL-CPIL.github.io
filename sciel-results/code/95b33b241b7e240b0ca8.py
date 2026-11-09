"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    # Construct the global optimum: a 100-bit binary string of all ones
    # This maximizes the number of ones (the LMX OneMax objective)
    # No search or deadline guard needed since this is O(1) and deterministic
    return "1" * N
# EVOLVE-BLOCK-END
