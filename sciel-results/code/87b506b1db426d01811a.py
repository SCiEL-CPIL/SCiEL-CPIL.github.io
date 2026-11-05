"""
Constructs a 2-coloring of K_n that avoids K_{2,2,2} in both colors.
Uses the Paley graph construction for n=29.
The Ramsey number R(K_{2,2,2}) is 30, so n=29 is the maximum possible.
"""

def construct():
    n = 29
    # Paley graph on Z_29
    # Vertices are 0..28
    # Edge (i, j) has color 0 if (j-i) % 29 is a quadratic residue
    # Edge (i, j) has color 1 otherwise
    
    # Precompute quadratic residues mod 29
    residues = set()
    for x in range(1, 29):
        residues.add((x * x) % 29)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in residues:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}
