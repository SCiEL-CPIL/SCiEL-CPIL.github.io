"""
Constructs a 12-state binary deterministic automaton that synchronizes.
This implementation uses the standard Černý automaton construction known to 
achieve the theoretical maximum reset word length of (N-1)^2 for N=12, 
which is 121.
"""

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Letter 'a': a cycle 0 -> 1 -> ... -> N-1 -> 0
    a = [(i + 1) % N for i in range(N)]
    
    # Letter 'b': maps 0 to 0, and i to (i+1)%N for i > 0
    # This specific configuration is known to yield the maximum reset threshold.
    b = [0] * N
    for i in range(1, N):
        b[i] = (i + 1) % N
        
    return [a, b]
# EVOLVE-BLOCK-END
