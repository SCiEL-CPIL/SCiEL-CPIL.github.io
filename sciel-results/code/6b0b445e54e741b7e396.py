import itertools

N = 12

def construct():
    # Pinwheel construction for N=12 states, binary alphabet {a, b}
    # This is the known optimal family for Černý's conjecture.
    # For N states, the pinwheel has reset threshold (N-1)*(N-2)/2
    # which for N=12 gives 55. However, there exist constructions
    # that achieve higher values.
    
    # Let's use the standard pinwheel:
    # a: i -> (i+1) % N (cyclic permutation)
    # b: 0 -> 0, i -> i-1 for i >= 1 (fixes 0, shifts others down)
    
    a = [(i + 1) % N for i in range(N)]
    b = [0] + [i - 1 for i in range(1, N)]
    
    return [a, b]
