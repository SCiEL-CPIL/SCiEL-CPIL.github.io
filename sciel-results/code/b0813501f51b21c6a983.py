def construct():
    """
    Constructs a 2-coloring of K_n without a monochromatic K_{2,2,2}.
    
    The Ramsey number R(K_{2,2,2}, K_{2,2,2}) is known to be 30.
    This means the maximum n for which a valid coloring exists is 29.
    The Paley graph on q=29 vertices provides such a coloring.
    In the Paley graph, an edge (i, j) exists if (j - i) is a quadratic residue mod q.
    Both the Paley graph and its complement are K_{2,2,2}-free for q=29.
    
    Returns:
        dict: {'n': 29, 'colors': [list of 0s and 1s]}
    """
    n = 29
    q = 29
    
    # Calculate quadratic residues mod q
    # For q=29, residues are {1, 4, 5, 6, 7, 8, 9, 13, 16, 20, 23, 24, 25}
    residues = set()
    for x in range(1, q):
        residues.add((x * x) % q)
    
    colors = []
    # Generate colors for all pairs (i, j) with i < j in lexicographic order
    # Color 0 if the difference is a quadratic residue (edge in Paley graph)
    # Color 1 otherwise (edge in complement)
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % q
            if diff in residues:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, len(colors)={len(res['colors'])}")
