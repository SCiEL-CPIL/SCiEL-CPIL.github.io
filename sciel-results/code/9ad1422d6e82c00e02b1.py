"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    n = 128
    # Start with all odds (safe), then try adding evens greedily
    S = set(range(1, n + 1, 2))  # all odds, size 64
    
    # Try adding even numbers in descending order
    for e in range(n, 0, -2):
        # Check if adding e creates a violation
        valid = True
        S_test = S | {e}
        for a, b, c in itertools.combinations(sorted(S_test), 3):
            if (a + b) in S_test and (a + c) in S_test and (b + c) in S_test:
                valid = False
                break
        if valid:
            S = S_test
    
    return sorted(S)
# EVOLVE-BLOCK-END
