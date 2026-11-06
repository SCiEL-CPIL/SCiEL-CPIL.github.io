"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
def construct():
    # Construct the Černý automaton, which is known to have a reset threshold
    # of N(N-1)/2 for N states. This is the best known lower bound for the
    # maximum threshold of binary automata.
    # Letter 0: Cycle 0 -> 1 -> ... -> N-1 -> 0
    # Letter 1: 0 -> 0, i -> i-1 for i > 0
    
    letter_a = [(i + 1) % N for i in range(N)]
    letter_b = [0 if i == 0 else i - 1 for i in range(N)]
    
    return [letter_a, letter_b]
# EVOLVE-BLOCK-END
