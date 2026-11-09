"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Paley tournament on 67 vertices: i -> j iff j - i is a nonzero square mod 67.
    # 67 is the smallest known n for which every 4-set has a common dominator.
    p = 67
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            # Check if j - i is a quadratic residue mod p
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
# EVOLVE-BLOCK-END
