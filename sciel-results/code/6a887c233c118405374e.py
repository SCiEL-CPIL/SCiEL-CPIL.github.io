def construct():
    """
    Construct a tournament on n vertices where every set of 4 vertices
    has a common dominator (a vertex with arcs to all four).
    
    Uses the Paley tournament on 67 vertices (p=67, p≡3 mod 4 prime),
    which is known to satisfy the condition that every 4-set has a
    common dominator. This gives f(4) <= 67.
    
    Returns:
        dict with 'n' (number of vertices) and 'orientation' (list of bits,
        one per pair (i,j) with i<j in lexicographic order; 0 means i->j,
        1 means j->i).
    """
    p = 67
    # Quadratic residues mod p (excluding 0)
    squares = {x * x % p for x in range(1, p)}
    # Build orientation: for each pair (i, j) with i < j,
    # 0 means i -> j, which happens when (j - i) % p is a quadratic residue
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
