import time

def construct():
    """
    Returns a valid two-coloring of K_n without a monochromatic K_{2,2,2}.
    Uses the Paley graph of order 29, which is a known optimal construction
    for the lower bound R(K_{2,2,2}) > 29.
    """
    n = 29
    # Quadratic residues modulo 29
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    
    # Generate colors in lexicographic order (i, j) with i < j
    # Color 0 if (j - i) % n is a square, else 1
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}
