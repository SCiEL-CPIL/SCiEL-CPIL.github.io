"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
def construct():
    # Pinwheel automaton: known to achieve (N-1)^2 reset length
    # a: cyclic shift
    # b: identity except 0 -> 1
    a = [(i + 1) % N for i in range(N)]
    b = [i if i != 0 else 1 for i in range(N)]
    return [a, b]
# EVOLVE-BLOCK-END
