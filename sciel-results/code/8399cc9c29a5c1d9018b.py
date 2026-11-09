"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Use the Paley tournament on 67 vertices.
    # 67 is a prime with 67 % 4 == 3.
    # The Paley tournament is defined by i -> j iff (j - i) % p is a quadratic residue.
    # It is a known result that the Paley tournament on 67 vertices satisfies the 
    # condition that every set of 4 vertices has a common dominator.
    
    p = 67
    # Compute quadratic residues modulo p
    squares = {x * x % p for x in range(1, p)}
    
    # Construct the orientation list
    # One bit per pair (i, j), i < j, in lexicographic order.
    # 0 means i -> j, 1 means j -> i.
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
                
    return {'n': p, 'orientation': orientation}
# EVOLVE-BLOCK-END
