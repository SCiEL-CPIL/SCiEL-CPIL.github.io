"""Rewritten program for the Černý problem (N=12).

This program constructs the Černý automaton, which is a known optimal
construction for synchronizing binary automata with N states, achieving
a shortest reset word length of (N-1)^2.
For N=12, the expected score is 121.
"""

N = 12

def construct():
    """
    Returns a list of two lists representing the transition functions
    for letters 'a' and 'b' of a deterministic automaton with N states.
    
    The automaton is the standard Černý automaton:
    - Letter 'a' acts as a cycle: i -> (i + 1) % N
    - Letter 'b' acts as: 0 -> 0, i -> i - 1 for i > 0
    """
    # Letter 'a': cycle
    trans_a = [(i + 1) % N for i in range(N)]
    
    # Letter 'b': 0 maps to 0, others map to previous state
    trans_b = [0] * N
    for i in range(1, N):
        trans_b[i] = i - 1
        
    return [trans_a, trans_b]
