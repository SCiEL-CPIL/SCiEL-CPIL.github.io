"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Use the Paley tournament on 67 vertices.
    # It is known that the Paley tournament for p=67 satisfies the Schütte condition for k=4.
    # This improves upon the seed's n=83.
    p = 67
    
    # Precompute squares modulo p
    squares = {x * x % p for x in range(1, p)}
    
    # Construct orientation bits
    # i -> j iff (j - i) % p is a square
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
                
    return {'n': p, 'orientation': orientation}
# EVOLVE-BLOCK-END
