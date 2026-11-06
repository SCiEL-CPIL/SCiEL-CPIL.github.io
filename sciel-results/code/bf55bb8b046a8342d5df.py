"""
Rewritten program to maximize n for the R(K2,2,2) task.
Uses the Paley graph construction for n=29, which is the known record.
"""

def construct():
    # The Paley graph of order 29 provides a valid 2-coloring of K_29
    # with no monochromatic K_{2,2,2}.
    # This certifies R(K_{2,2,2}) > 29.
    n = 29
    
    # Compute non-zero quadratic residues modulo n
    # A number x is a quadratic residue mod p if x = y^2 mod p for some y != 0
    residues = set()
    for x in range(1, n):
        residues.add((x * x) % n)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            # Color 0 if diff is a non-zero quadratic residue
            # Color 1 otherwise (which includes non-residues and 0, but diff != 0 here)
            if diff in residues:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}
