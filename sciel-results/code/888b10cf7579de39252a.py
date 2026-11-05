import itertools
import time

def construct():
    # The Schläfli graph is a strongly regular graph with parameters (35, 16, 7, 8).
    # It is K4-free (clique number 3) and has independence number 5.
    # Construction:
    # Vertices are the 3-element subsets of {0, 1, 2, 3, 4, 5, 6}.
    # Two vertices (subsets A and B) are adjacent if |A intersect B| is 2 or 0.
    # (They are NOT adjacent if |A intersect B| is 1).
    
    base_set = list(range(7))
    vertices = list(itertools.combinations(base_set, 3))
    n = len(vertices) # Should be 35
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            a = set(vertices[i])
            b = set(vertices[j])
            inter_size = len(a.intersection(b))
            if inter_size == 2 or inter_size == 0:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
