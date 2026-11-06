"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the classic Černý automaton for N=12 states.
    
    Letter 'a': cyclic shift forward: i -> (i+1) % N
    Letter 'b': i -> i if i==0, else i -> (i-1) % N
    
    This automaton synchronizes with shortest reset word of length (N-1)^2 = 121.
    The reset word is: b^(N-1) a b^(N-1) a ... b^(N-1) a (N-1 times the pattern b^(N-1)a)
    minus the final a, giving length (N-1)^2.
    """
    # Letter 'a': forward cycle
    a_trans = [(i + 1) % N for i in range(N)]
    
    # Letter 'b': 0 maps to 0, others map backward
    b_trans = [0] + [i - 1 for i in range(1, N)]
    
    return [a_trans, b_trans]
# EVOLVE-BLOCK-END
