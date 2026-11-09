"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    # Prioritize vectors with only 0s and 1s (binary vectors).
    # The set of all binary vectors in {0,1,2}^n is a cap set of size 2^n.
    # Vectors containing 2s are deprioritized.
    if 2 in vector:
        return -1.0
    # Among binary vectors, prioritize those with lower Hamming weight?
    # Or just return 0.0 to keep lexicographic order among them.
    return 0.0
# EVOLVE-BLOCK-END
