import time
import itertools
import random

def _is_2_colorable(edges, n):
    """Check if the hypergraph is 2-colorable."""
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

def _find_3_coloring(edges, n):
    """Find a 3-coloring if one exists, else None."""
    for mask in range(3 ** n):
        temp = mask
        colors = [0] * n
        valid = True
        for i in range(n):
            colors[i] = temp % 3
            temp //= 3
        for e in edges:
            if colors[e[0]] == colors[e[1]] == colors[e[2]]:
                valid = False
                break
        if valid:
            return colors
    return None

def _search_min_edges(n=9, deadline=85.0):
    """Search for minimum edge count 3-chromatic hypergraph."""
    all_edges = list(itertools.combinations(range(n), 3))
    
    for k in range(1, 11):
        if time.time() > deadline:
            break
        
        if k <= 5:
            for edge_set in itertools.combinations(all_edges, k):
                edges = list(edge_set)
                if _is_2_colorable(edges, n):
                    continue
                colors = _find_3_coloring(edges, n)
                if colors is not None:
                    return edges, colors
        elif k <= 7:
            for _ in range(500000):
                if time.time() > deadline:
                    break
                edges = random.sample(all_edges, k)
                if _is_2_colorable(edges, n):
                    continue
                colors = _find_3_coloring(edges, n)
                if colors is not None:
                    return edges, colors
        else:
            for _ in range(1000000):
                if time.time() > deadline:
                    break
                edges = random.sample(all_edges, k)
                if _is_2_colorable(edges, n):
                    continue
                colors = _find_3_coloring(edges, n)
                if colors is not None:
                    return edges, colors
    
    return None, None

def construct():
    start_time = time.time()
    deadline = 85.0
    
    edges, colors = _search_min_edges(9, deadline)
    
    if edges is not None:
        return {
            'edges': [list(e) for e in edges],
            'colors': colors
        }
    
    # Fallback: Fano plane (known 3-chromatic with 7 edges)
    fano_edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    fano_colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {
        'edges': fano_edges,
        'colors': fano_colors
    }
