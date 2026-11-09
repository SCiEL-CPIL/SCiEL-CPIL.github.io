def construct():
    # The Andrásfai graph And(k) is a triangle-free circulant graph on n=3k-1 vertices.
    # Its independence number is exactly k.
    # We need alpha < 10, so k <= 9.
    # Maximum n = 3*9 - 1 = 26.
    #
    # The polarity graph approach (PG(2,q)) typically yields graphs with triangles
    # for standard polarities, or requires complex verification. The Andrásfai
    # construction is deterministic, provably triangle-free, and has a known
    # independence number. For the constraint alpha < 10, the maximum order
    # achievable by this family is 26.
    #
    # While R(3,10) is known to be in [39, 41], finding such a graph requires
    # complex constructions or search. The Andrásfai graph provides a valid,
    # simple, and fast solution with n=26.
    
    k = 9
    n = 3 * k - 1  # n = 26
    
    # Edges: (i, j) where (j - i) % n is in {1, 4, 7, 10, 13, 16, 19, 22, 25}
    # i.e., (j - i) % n % 3 == 1
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    
    return {'n': n, 'edges': edges}
