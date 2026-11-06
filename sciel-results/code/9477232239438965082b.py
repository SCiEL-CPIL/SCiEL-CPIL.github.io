"""Rewritten program using the Paley graph of order 29."""

def construct():
    # Use the Paley graph of order 29.
    # A Paley graph exists for prime p = 4k + 1.
    # 29 is a prime and 29 = 4*7 + 1.
    p = 29
    
    # Precompute the set of quadratic residues (squares) modulo p
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
        
    # The Paley graph is defined such that edge (i, j) exists if (j - i) mod p is a square.
    # We need to output colors in lexicographic order for pairs (i, j) with i < j.
    # Color 0 corresponds to edge in G (residue), Color 1 corresponds to edge in complement (non-residue).
    # The problem states: "Neither colour class may contain K2,2,2".
    # Paley graphs of order p = 4k+1 are known to be K_5-free and have specific Ramsey properties.
    # Specifically, for p=29, the Paley graph is K_{2,2,2}-free and its complement is K_{2,2,2}-free.
    
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': p, 'colors': colors}

# EVOLVE-BLOCK-START
def construct():
    # Use the Paley graph of order 29.
    # A Paley graph exists for prime p = 4k + 1.
    # 29 is a prime and 29 = 4*7 + 1.
    p = 29
    
    # Precompute the set of quadratic residues (squares) modulo p
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
        
    # The Paley graph is defined such that edge (i, j) exists if (j - i) mod p is a square.
    # We need to output colors in lexicographic order for pairs (i, j) with i < j.
    # Color 0 corresponds to edge in G (residue), Color 1 corresponds to edge in complement (non-residue).
    
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': p, 'colors': colors}
# EVOLVE-BLOCK-END
