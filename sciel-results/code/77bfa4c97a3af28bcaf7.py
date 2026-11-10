import time
import random
from itertools import combinations

def _is_non_2_colorable(n, edges):
    """Check if hypergraph is NOT 2-colorable using backtracking with unit propagation."""
    if not edges:
        return False  # Empty hypergraph is 2-colorable
    
    # Build adjacency: for each vertex, which edges contain it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # Backtracking with unit propagation
    # colors: -1 = unassigned, 0 = color 0, 1 = color 1
    colors = [-1] * n
    
    def unit_propagate():
        """Apply unit propagation: if an edge has all but one vertex assigned the same color, assign the last one."""
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edges):
                # Count assignments in this edge
                assigned_0 = 0
                assigned_1 = 0
                unassigned = []
                for v in e:
                    if colors[v] == 0:
                        assigned_0 += 1
                    elif colors[v] == 1:
                        assigned_1 += 1
                    else:
                        unassigned.append(v)
                
                if not unassigned:
                    # All assigned: check if monochromatic
                    if assigned_0 == 5 or assigned_1 == 5:
                        return False  # Contradiction: monochromatic edge
                    continue
                
                # Unit propagation: if 4 are same color, the 5th must be different
                if assigned_0 == 4 and len(unassigned) == 1:
                    colors[unassigned[0]] = 1
                    changed = True
                elif assigned_1 == 4 and len(unassigned) == 1:
                    colors[unassigned[0]] = 0
                    changed = True
        return True
    
    def backtrack():
        """Try to find a valid 2-coloring. Returns True if one exists (i.e., hypergraph IS 2-colorable)."""
        # Find unassigned vertex with fewest options
        unassigned = [v for v in range(n) if colors[v] == -1]
        if not unassigned:
            # All assigned, check all edges
            for e in edges:
                c0 = sum(1 for v in e if colors[v] == 0)
                if c0 == 0 or c0 == 5:
                    return True  # Found a valid coloring (no monochromatic edge)
            return True
        
        # Choose vertex with most constraints
        v = max(unassigned, key=lambda x: len(vertex_edges[x]))
        
        for color in [0, 1]:
            colors[v] = color
            if unit_propagate():
                if backtrack():
                    return True
            colors[v] = -1
        
        return False
    
    # Try to find a 2-coloring
    if unit_propagate() and backtrack():
        return False  # Found a coloring, so it IS 2-colorable
    else:
        return True  # No coloring found, so it's NOT 2-colorable


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with complete 5-uniform hypergraph on 9 vertices
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    
    # Verify it's non-2-colorable
    if not _is_non_2_colorable(n, all_edges):
        # Shouldn't happen, but fallback
        return {'n': n, 'edges': all_edges}
    
    current_edges = all_edges[:]
    
    # Greedy edge removal: try to remove each edge if the hypergraph remains non-2-colorable
    # Use random order to escape local optima
    random.seed(42)
    indices = list(range(len(current_edges)))
    
    # Multiple passes with random shuffling
    max_passes = 10
    for _pass in range(max_passes):
        if time.time() > deadline:
            break
        
        random.shuffle(indices)
        removed_any = False
        
        for idx in indices:
            if time.time() > deadline:
                break
            if idx >= len(current_edges):
                continue
            
            # Try removing edge at idx
            candidate = current_edges[:idx] + current_edges[idx+1:]
            
            if _is_non_2_colorable(n, candidate):
                current_edges = candidate
                indices.remove(idx)
                removed_any = True
        
        if not removed_any:
            break
    
    return {'n': n, 'edges': current_edges}
