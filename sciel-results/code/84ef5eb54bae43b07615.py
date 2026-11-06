"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
def construct():
    # Standard Černý automaton construction for N states.
    # a: cycle 0->1->...->N-1->0
    # b: maps all states to 0, except 0 maps to 1
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    b[0] = 1
    return [a, b]
# EVOLVE-BLOCK-END
