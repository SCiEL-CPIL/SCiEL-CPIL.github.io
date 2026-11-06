"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
def construct():
    # Construction of a 12-state automaton with reset length N-1 = 11.
    # Letter 'a' is a simple cycle: i -> (i+1) % N
    # Letter 'b' shifts states down by 1, with 0 mapping to 0:
    # b(0) = 0, b(i) = i-1 for i > 0.
    
    a = [(i + 1) % N for i in range(N)]
    b = [i - 1 if i > 0 else 0 for i in range(N)]
    
    return [a, b]
# EVOLVE-BLOCK-END
