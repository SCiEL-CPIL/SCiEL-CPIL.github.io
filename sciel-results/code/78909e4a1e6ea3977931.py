import time
import random
from itertools import combinations

def _is_non_2colorable_fast(n, edges):
    """
    Check if the hypergraph is NOT 2-colorable by enumerating all 2^n colorings.
    Returns True if no valid 2-coloring exists.
    """
    m = len(edges)
    if m == 0:
        return False
    
    # Precompute edge vertex lists
    edge_verts = [list(e) for e in edges]
    
    # Iterate through all 2^n colorings
    # We can prune if we find one valid coloring
    for mask in range(1 << n):
        # Check if this coloring is valid
        valid = True
        for e in edge_verts:
            # Check if edge e is monochromatic
            # Get colors of vertices in e
            # If all 0 or all 1, it's monochromatic
            is_mono = True
            first_color = -1
            for v in e:
                c = (mask >> v) & 1
                if first_color == -1:
                    first_color = c
                elif c != first_color:
                    is_mono = False
                    break
            if is_mono:
                valid = False
                break
        if valid:
            return False # Found a valid coloring, so it IS 2-colorable
            
    return True # No valid coloring found, so it is NOT 2-colorable

def _greedy_delete_fast(n, edges, deadline):
    """Greedy edge deletion using the fast checker."""
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        i = 0
        while i < len(edges) and time.time() < deadline:
            if time.time() > deadline:
                break
            # Try removing edge i
            candidate = edges[:i] + edges[i+1:]
            if _is_non_2colorable_fast(n, candidate):
                edges.pop(i)
                removed_any = True
            else:
                i += 1
    return edges

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin
    
    best_m = float('inf')
    best_graph = None
    
    # Try n=9 first (K9^5 is non-2-colorable)
    for n in [9, 10, 11]:
        if time.time() > deadline - 5:
            break
            
        all_edges = [list(e) for e in combinations(range(n), 5)]
        
        # Check if complete hypergraph is non-2-colorable
        if not _is_non_2colorable_fast(n, all_edges):
            continue # It is 2-colorable, skip this n
            
        # Greedy deletion
        edges = all_edges[:]
        random.shuffle(edges)
        
        edges = _greedy_delete_fast(n, edges, deadline - 2)
        
        if len(edges) < best_m:
            best_m = len(edges)
            best_graph = {'n': n, 'edges': edges}
            
        # If we found a very good one, we can stop
        if best_m <= 51:
            break

    # If we didn't find anything (shouldn't happen for n=9), fallback
    if best_graph is None:
        # Fallback to K9^5
        n = 9
        edges = [list(e) for e in combinations(range(n), 5)]
        best_graph = {'n': n, 'edges': edges}

    return best_graph
