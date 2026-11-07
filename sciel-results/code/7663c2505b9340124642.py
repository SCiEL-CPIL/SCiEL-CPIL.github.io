import random
from itertools import combinations

def is_2_colorable(edges):
    """Check if hypergraph admits a 2-coloring with no monochromatic edges."""
    n = 9
    for mask in range(1 << n):
        valid = True
        for e in edges:
            c0 = (mask >> e[0]) & 1
            c1 = (mask >> e[1]) & 1
            c2 = (mask >> e[2]) & 1
            if c0 == c1 == c2:
                valid = False
                break
        if valid:
            return True, None
    return False, None

def find_3_coloring(edges):
    """Find a 3-coloring with no monochromatic edges, or return None."""
    n = 9
    for mask in range(3 ** n):
        valid = True
        for e in edges:
            c0 = (mask // (3 ** e[0])) % 3
            c1 = (mask // (3 ** e[1])) % 3
            c2 = (mask // (3 ** e[2])) % 3
            if c0 == c1 == c2:
                valid = False
                break
        if valid:
            return [(mask // (3 ** i)) % 3 for i in range(n)]
    return None

def is_3_chromatic(edges):
    """Check if hypergraph is exactly 3-chromatic (not 2-colorable but 3-colorable)."""
    two_colorable, _ = is_2_colorable(edges)
    if two_colorable:
        return False, None
    coloring = find_3_coloring(edges)
    if coloring is None:
        return False, None
    return True, coloring

def construct():
    """Construct a 3-chromatic triple hypergraph on 9 vertices with minimum edges."""
    
    # All possible triples on 9 vertices
    all_triples = list(combinations(range(9), 3))
    
    # Try edge counts from small to large
    for num_edges in range(3, 12):
        # Random search for this edge count
        random.seed(42)
        for attempt in range(100000):
            # Sample random distinct triples
            edges = random.sample(all_triples, num_edges)
            is_chromatic, coloring = is_3_chromatic(edges)
            if is_chromatic:
                return {
                    'edges': [list(e) for e in edges],
                    'colors': [int(c) for c in coloring]
                }
    
    # Fallback: known 10-edge solution (complete 3-uniform on 5 vertices)
    edges_10 = [list(e) for e in combinations(range(5), 3)]
    # Find a valid 3-coloring for this
    coloring = find_3_coloring(edges_10)
    if coloring is None:
        # Should not happen, but provide a default
        coloring = [0, 1, 2, 0, 1, 0, 0, 0, 0]
    
    return {
        'edges': edges_10,
        'colors': [int(c) for c in coloring]
    }
