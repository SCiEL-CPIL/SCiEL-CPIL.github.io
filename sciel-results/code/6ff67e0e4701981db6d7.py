"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Maximum number of MOLS of order 10 is 1 (Tarry 1901).
    # Return a single cyclic Latin square of order 10.
    n = 10
    square = [[(r + c) % n for c in range(n)] for r in range(n)]
    return [square]
# EVOLVE-BLOCK-END
