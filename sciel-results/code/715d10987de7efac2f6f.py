N = 12

# EVOLVE-BLOCK-START
def construct():
    # Alternative construction: 
    # State 0 is the "sink" for letter 'b'.
    # Letter 'a' is a cyclic permutation.
    # Letter 'b' maps 0->0 and i->i-1.
    # This is isomorphic to the standard Černý automaton.
    a = []
    b = []
    for i in range(N):
        # a: cycle 0->1->...->N-1->0
        a.append((i + 1) % N)
        # b: 0->0, i->i-1
        if i == 0:
            b.append(0)
        else:
            b.append(i - 1)
    return [a, b]
# EVOLVE-BLOCK-END
