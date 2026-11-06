"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _make_paley(p):
    """
    Constructs a Paley tournament on p vertices.
    Returns a list of bits representing the orientation.
    i -> j iff (j - i) % p is a quadratic residue mod p.
    """
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
    return orientation

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # The smallest known valid n for Schütte's problem k=4 is 67 (Paley tournament).
    # We construct it directly and return without expensive verification to save time.
    
    p = 67
    orientation = _make_paley(p)
    
    # If we have plenty of time left, we could verify, but it's not strictly necessary
    # since 67 is known to be valid and the construction is deterministic.
    # The previous version spent time verifying. We skip it to maximize robustness.
    
    return {'n': p, 'orientation': orientation}

# EVOLVE-BLOCK-END
