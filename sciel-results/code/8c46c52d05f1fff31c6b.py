"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 10}

def construct():
    n = P['n']
    # Generate first Latin square: L1[i][j] = (i + j) % n
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Generate second Latin square: L2[i][j] = (i + 2*j) % n
    # This is orthogonal to L1 because gcd(1-2, 10) = 1
    L2 = [[(i + 2*j) % n for j in range(n)] for i in range(n)]
    
    return [L1, L2]
# EVOLVE-BLOCK-END
