"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Paley tournament on 67 vertices.
    # The Paley tournament on p vertices (p prime, p = 3 mod 4) is defined by:
    # i -> j iff (j - i) mod p is a quadratic residue mod p.
    # It is known that the Paley tournament on 67 vertices satisfies the condition
    # that every set of 4 vertices has a common dominator.
    # This improves upon the seed program's 83 vertices.
    p = 67
    squares = {x * x % p for x in range(1, p)}
    return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}
# EVOLVE-BLOCK-END
