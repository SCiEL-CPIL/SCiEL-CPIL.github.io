"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a tournament on n vertices where every set of 4 vertices has a common dominator.
    Uses the Paley tournament on p=67 vertices, which is known to satisfy this property
    and is the smallest known such construction.
    """
    p = 67
    
    # Construct quadratic residues mod p
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    
    # Construct orientation: one bit per pair (i,j) with i<j in lexicographic order
    # 0 means i -> j, 1 means j -> i
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)  # i -> j
            else:
                orientation.append(1)  # j -> i
                
    return {'n': p, 'orientation': orientation}

# EVOLVE-BLOCK-END
