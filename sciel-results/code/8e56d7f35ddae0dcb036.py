"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import zlib

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Use a deterministic hash to randomize the order of vector selection.
    # This avoids the suboptimal structure of the {0,1}^n hypercube
    # and allows the greedy algorithm to explore a wider variety of
    # cap set configurations, which typically yields larger sets
    # for n >= 4.
    return float(zlib.crc32(bytes(vector)))
# EVOLVE-BLOCK-END
