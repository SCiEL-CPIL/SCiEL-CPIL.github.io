"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the Černý automaton for N=12.
    
    The Černý automaton is a deterministic finite automaton with N states and 2 letters
    that achieves the maximum possible reset word length of (N-1)^2 = 121 for N=12.
    
    The automaton is defined as:
    - States: 0, 1, ..., N-1
    - Letter 'a': (i+1) % N  (cycle forward)
    - Letter 'b': max(i-1, 0) (shift toward 0, with 0 fixed)
    
    This is the standard Černý automaton construction which provably has reset threshold
    exactly (N-1)^2 for all N >= 2.
    """
    # Letter 'a': cyclic shift forward
    letter_a = [(i + 1) % N for i in range(N)]
    # Letter 'b': shift toward 0 (0 is fixed, others move one step toward 0)
    letter_b = [max(i - 1, 0) for i in range(N)]
    
    return [letter_a, letter_b]
# EVOLVE-BLOCK-END
