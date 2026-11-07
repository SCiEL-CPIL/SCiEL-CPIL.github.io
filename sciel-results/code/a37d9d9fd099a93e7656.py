import time
import itertools
import random

def _is_2_colorable(edges):
    """Check if hypergraph admits a 2-coloring (no monochromatic edge)."""
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
            return True
    return False

def _find_3_coloring(edges):
    """Find a 3-coloring where no edge is monochromatic."""
    n = 9
    for mask in range(3 ** n):
        m = mask
        colors = [0] * n
        for i in range(n):
            colors[i] = m % 3
            m //= 3
        valid = True
        for e in edges:
            c0 = colors[e[0]]
            c1 = colors[e[1]]
            c2 = colors[e[2]]
            if c0 == c1 == c2:
                valid = False
                break
        if valid:
            return colors
    return None

def construct():
    deadline = time.time() + 75  # safety margin before 90s kill
    
    # Known 7-edge solution (Fano plane embedded in 9 vertices)
    fano_edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    fano_colors = [0, 0, 1, 1, 2, 1, 2, 0, 0]
    
    # Try to find a 6-edge 3-chromatic hypergraph
    # Use randomized search with local improvement
    all_triples = list(itertools.combinations(range(9), 3))
    random.seed(42)
    
    best_edges = None
    best_colors = None
    
    # Try many random 6-edge subsets
    max_attempts = 50000
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Random 6-edge subset
        subset = random.sample(all_triples, 6)
        
        # Quick check: must not be 2-colorable
        if _is_2_colorable(subset):
            continue
        
        # Check if 3-colorable
        colors = _find_3_coloring(subset)
        if colors is not None:
            best_edges = [list(e) for e in subset]
            best_colors = colors
            break
    
    if best_edges is not None:
        return {
            'edges': best_edges,
            'colors': best_colors
        }
    
    # Fall back to 7-edge Fano plane
    return {
        'edges': fano_edges,
        'colors': fano_colors
    }
