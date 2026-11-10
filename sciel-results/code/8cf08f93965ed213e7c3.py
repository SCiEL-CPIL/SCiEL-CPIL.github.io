"""Find a small 5-uniform hypergraph without Property B (no proper 2-colouring)."""
import time
import random
from itertools import combinations

def _is_2_colorable(n, edges):
    """Exact check via backtracking with unit propagation (bitmask-based)."""
    if not edges:
        return True
    
    # Precompute bitmask for each edge
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # For each edge, precompute the list of vertices
    edge_verts = [list(e) for e in edges]
    
    # We'll use a recursive backtracking with unit propagation
    # State: color assignment (None = uncolored), represented as dict or array
    # Colors: 0 and 1
    
    colors = [None] * n
    edge_satisfied = [False] * len(edges)  # edge is "satisfied" (has both colors) or "violated" (monochromatic)
    # Actually for Property B: an edge is violated if all its vertices have the same color.
    # We need NO edge to be violated at the end.
    
    # Better approach: track for each edge how many of each color are present.
    # edge_color_count[edge_idx] = [count_color0, count_color1]
    # An edge is violated if count_color0 == 0 (all color 1) or count_color1 == 0 (all color 0)
    # and all vertices are colored.
    
    # Unit propagation: if an edge has k-1 vertices of color c and the last one is uncolored,
    # then the last one MUST be color 1-c to avoid monochromatic.
    
    # Let's implement a cleaner version.
    
    def propagate(colors, edge_color_count):
        """Apply unit propagation. Returns False if contradiction, True otherwise."""
        changed = True
        while changed:
            changed = False
            for ei, e in enumerate(edge_verts):
                c0, c1 = edge_color_count[ei]
                uncolored = [v for v in e if colors[v] is None]
                if not uncolored:
                    # Edge fully colored
                    if c0 == 0 or c1 == 0:
                        return False  # Monochromatic edge
                    continue
                # If all colored vertices have same color, force the uncolored ones to be the other color
                if c0 > 0 and c1 == 0:
                    # All colored are color 0, so uncolored must be color 1
                    for v in uncolored:
                        if colors[v] is not None:
                            if colors[v] == 0:
                                return False
                            continue
                        colors[v] = 1
                        edge_color_count[ei][1] += 1
                        changed = True
                elif c1 > 0 and c0 == 0:
                    # All colored are color 1, so uncolored must be color 0
                    for v in uncolored:
                        if colors[v] is not None:
                            if colors[v] == 1:
                                return False
                            continue
                        colors[v] = 0
                        edge_color_count[ei][0] += 1
                        changed = True
        return True
    
    # We need to track edge_color_count properly. Let me restructure.
    
    def check_2colorable(n, edge_verts, edge_masks):
        m = len(edge_verts)
        if m == 0:
            return True
        
        # Use a simpler backtracking without complex propagation for speed
        # with good variable ordering
        
        # Order edges by some heuristic
        # Order vertices by degree
        deg = [0] * n
        for e in edge_verts:
            for v in e:
                deg[v] += 1
        vertex_order = sorted(range(n), key=lambda v: -deg[v])
        
        # Try assignments in order of vertex_order
        # Use recursion with memoization-like pruning
        
        import sys
        sys.setrecursionlimit(10000)
        
        def backtrack(idx, colors):
            if idx == n:
                # Check all edges
                for e in edge_verts:
                    colors_in_edge = [colors[v] for v in e]
                    if all(c == colors_in_edge[0] for c in colors_in_edge):
                        return False
                return True
            
            v = vertex_order[idx]
            
            # Try color 0
            colors[v] = 0
            # Check if any edge becomes monochromatic with this assignment
            # (i.e., all other vertices in the edge are also color 0)
            valid = True
            for e in edge_verts:
                if v in e:
                    others = [colors[u] for u in e if u != v]
                    if all(c == 0 for c in others):
                        valid = False
                        break
            if valid:
                if backtrack(idx + 1, colors):
                    return True
            
            # Try color 1
            colors[v] = 1
            valid = True
            for e in edge_verts:
                if v in e:
                    others = [colors[u] for u in e if u != v]
                    if all(c == 1 for c in others):
                        valid = False
                        break
            if valid:
                if backtrack(idx + 1, colors):
                    return True
            
            colors[v] = None
            return False
        
        colors = [None] * n
        return backtrack(0, colors)
    
    return check_2colorable(n, edge_verts, edge_masks)


def _try_remove_edge(n, edges, edge_idx, deadline):
    """Try removing edge at edge_idx. Returns True if still non-2-colorable."""
    if time.time() > deadline:
        return False
    new_edges = edges[:edge_idx] + edges[edge_idx+1:]
    return not _is_2_colorable(n, new_edges)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with complete 5-uniform hypergraph on 10 vertices
    # C(10,5) = 252 edges
    n = 10
    all_edges = [list(e) for e in combinations(range(n), 5)]
    edges = all_edges[:]
    
    # Deterministic greedy edge removal
    # Try to remove edges in a smart order
    # First pass: try each edge, remove if possible
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        i = 0
        while i < len(edges) and time.time() < deadline:
            if _try_remove_edge(n, edges, i, deadline):
                edges.pop(i)
                removed_any = True
            else:
                i += 1
    
    current_best = len(edges)
    current_best_edges = edges[:]
    
    # Try random restarts with different initial orderings
    # Or try adding/removing vertices
    
    # Try n=9 as well
    n9 = 9
    all_edges9 = [list(e) for e in combinations(range(n9), 5)]
    edges9 = all_edges9[:]
    
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        i = 0
        while i < len(edges9) and time.time() < deadline:
            if _try_remove_edge(n9, edges9, i, deadline):
                edges9.pop(i)
                removed_any = True
            else:
                i += 1
    
    if len(edges9) < current_best:
        current_best = len(edges9)
        current_best_edges = edges9[:]
        n = n9
    
    # Try random permutations of edge order for the removal process
    if time.time() < deadline - 10:
        for n_try in [10, 9, 11]:
            if time.time() > deadline - 5:
                break
            all_e = [list(e) for e in combinations(range(n_try), 5)]
            for trial in range(5):
                if time.time() > deadline - 5:
                    break
                random.shuffle(all_e)
                e_copy = all_e[:]
                removed_any = True
                while removed_any and time.time() < deadline - 5:
                    removed_any = False
                    i = 0
                    while i < len(e_copy) and time.time() < deadline - 5:
                        if _try_remove_edge(n_try, e_copy, i, deadline):
                            e_copy.pop(i)
                            removed_any = True
                        else:
                            i += 1
                if len(e_copy) < current_best:
                    current_best = len(e_copy)
                    current_best_edges = e_copy[:]
                    n = n_try
    
    # Final verification
    if _is_2_colorable(n, current_best_edges):
        # Fallback to complete graph on 9 vertices
        n = 9
        current_best_edges = [list(e) for e in combinations(range(n), 5)]
    
    return {'n': n, 'edges': current_best_edges}
