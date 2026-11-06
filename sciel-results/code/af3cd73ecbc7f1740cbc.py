import itertools

def construct():
    """
    Construct the Shrikhande graph, which is isomorphic to the line graph of K_{4,4}.
    It has 16 vertices and 48 edges.
    It is K_{4,4}-free, and thus Q3-free.
    """
    edges = []
    # Vertices are (i, j) for i, j in 0..3
    # Adjacency: (i, j) ~ (k, l) if i == k or j == l (and (i,j) != (k,l))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    if (i, j) < (k, l):
                        if i == k or j == l:
                            # Map (i, j) to vertex id i*4 + j
                            u = i * 4 + j
                            v = k * 4 + l
                            edges.append([u, v])
    return edges
