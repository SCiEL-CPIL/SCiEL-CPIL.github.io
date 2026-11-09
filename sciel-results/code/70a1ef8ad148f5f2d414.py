"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, permutations

# Precompute all 60 distinct 6-cycles on 6 labeled vertices
def _precompute_cycles():
    cycles = []
    seen = set()
    for perm in permutations(range(1, 6)):
        cycle = (0,) + perm
        edges = frozenset(frozenset((cycle[i], cycle[(i+1)%6])) for i in range(6))
        if edges not in seen:
            seen.add(edges)
            cycles.append(cycle)
    return cycles

_CYCLES = _precompute_cycles()

def _edge_index(i, j, n):
    """Get the index in the lexicographic list for edge (i,j) with i<j."""
    if i > j:
        i, j = j, i
    return i*(n-1) - i*(i-1)//2 + (j - i - 1)

def _has_c6_in_color(adj, n, color):
    """Check if a specific color class contains a 6-cycle."""
    if n < 6:
        return False
    
    # adj is a set of frozensets of edges for this color
    for subset in combinations(range(n), 6):
        for cycle in _CYCLES:
            ok = True
            for k in range(6):
                v = subset[cycle[k]]
                w = subset[cycle[(k+1)%6]]
                edge = frozenset((v, w))
                if edge not in adj:
                    ok = False
                    break
            if ok:
                return True
    return False

def _check_partial_validity(colors_so_far, n_current, n_target, deadline):
    """
    Check if the partial coloring (vertices 0..n_current-1) has no C6 in any color class.
    colors_so_far is a list of colors for edges among vertices 0..n_current-1.
    Returns True if valid so far.
    """
    if n_current < 6:
        return True
    
    # Build adjacency sets for each color
    color_edges = [set() for _ in range(4)]
    idx = 0
    for i in range(n_current):
        for j in range(i+1, n_current):
            c = colors_so_far[idx]
            color_edges[c].add(frozenset((i, j)))
            idx += 1
    
    for c in range(4):
        if _has_c6_in_color(color_edges[c], n_current, c):
            return False
    return True

def _backtrack_search(n_target, deadline):
    """
    Backtracking search to find a valid 4-coloring of K_n_target.
    Returns (colors, n) if found, else (None, None).
    """
    # We'll build the coloring vertex by vertex.
    # colors[i] will store the color of edge (i, j) for j > i
    # Actually, let's use a 2D array for clarity, then flatten at the end.
    
    # colors[i][j] for i < j
    colors = [[None]*n_target for _ in range(n_target)]
    
    # We assign colors for edges (v, w) where v < w, in lexicographic order of (v, w)
    # But we process vertex by vertex: when adding vertex v, we assign colors for (0,v), (1,v), ..., (v-1,v)
    
    def _search(v):
        """Try to assign colors for edges from v to all previous vertices."""
        if time.time() > deadline:
            return False
        
        if v == n_target:
            return True
        
        # Assign colors for edges (0,v), (1,v), ..., (v-1,v)
        # There are v edges to assign
        if v == 0:
            # No edges to assign
            return _search(v + 1)
        
        # Try all combinations of colors for the v edges
        # This is 4^v possibilities, which grows quickly
        # For v up to ~10, 4^10 = 1M, which is too much
        # We need pruning
        
        # Use iterative deepening with pruning
        def _assign(idx, colors_so_far):
            if time.time() > deadline:
                return False
            
            if idx == v:
                # All v edges assigned, check validity
                # Temporarily set the colors
                for i in range(v):
                    colors[i][v] = colors_so_far[i]
                
                # Check if adding vertex v creates a C6 in any color class
                # We need to check all 6-subsets that include v
                # Actually, we need to check if the current partial coloring has any C6
                # A C6 can only be formed if it includes the new vertex v
                # So we check all 6-subsets containing v
                
                if v >= 5:  # Need at least 6 vertices
                    # Check all 6-subsets containing v
                    # The other 5 vertices are chosen from 0..v-1
                    for others in combinations(range(v), 5):
                        subset = tuple(sorted(others + (v,)))
                        # Check if this subset contains a C6 in any color
                        for c in range(4):
                            ok = True
                            for cycle in _CYCLES:
                                all_edges_ok = True
                                for k in range(6):
                                    a = subset[cycle[k]]
                                    b = subset[cycle[(k+1)%6]]
                                    if a > b:
                                        a, b = b, a
                                    if colors[a][b] != c:
                                        all_edges_ok = False
                                        break
                                if all_edges_ok:
                                    # Found a C6 in color c
                                    # Undo and return False
                                    for i in range(v):
                                        colors[i][v] = None
                                    return False
                            # If we get here, no C6 in this subset for color c
                        # Continue to next subset
                
                # No C6 found, recurse
                if _search(v + 1):
                    return True
                
                # Undo
                for i in range(v):
                    colors[i][v] = None
                return False
            
            # Try each color for edge (idx, v)
            for c in range(4):
                colors_so_far[idx] = c
                if _assign(idx + 1, colors_so_far):
                    return True
            colors_so_far[idx] = None
            return False
        
        colors_so_far = [None] * v
        return _assign(0, colors_so_far)
    
    return _search(0)

def _flatten_colors(colors_2d, n):
    """Flatten 2D color array to lexicographic list."""
    result = []
    for i in range(n):
        for j in range(i+1, n):
            result.append(colors_2d[i][j])
    return result

def _is_valid_full(colors, n):
    """Check if a complete coloring is valid."""
    if n < 6:
        return True
    
    color_edges = [set() for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            color_edges[c].add(frozenset((i, j)))
            idx += 1
    
    for c in range(4):
        if _has_c6_in_color(color_edges[c], n, c):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with a known small valid coloring
    # K1 is trivially valid
    best_n = 1
    best_colors = []
    
    # Try to find valid colorings for increasing n
    # Use backtracking search with time limit
    for target_n in range(2, 21):
        if time.time() > deadline - 10:
            break
        
        # Try backtracking search
        found = _backtrack_search(target_n, deadline - 5)
        if found:
            pass
        
        # Actually, let me use a different approach:
        # Use local search with better initialization
        
        # Try multiple random restarts with local search
        for restart in range(20):
            if time.time() > deadline - 2:
                break
            
            # Generate random coloring
            m = target_n * (target_n - 1) // 2
            colors = np.random.randint(0, 4, size=m).tolist()
            
            # Local search: iteratively fix C6 violations
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                # Check validity
                if _is_valid_full(colors, target_n):
                    best_n = target_n
                    best_colors = colors
                    break
                
                # Find a C6 and fix it
                color_edges = [set() for _ in range(4)]
                idx = 0
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        c = colors[idx]
                        color_edges[c].add(frozenset((i, j)))
                        idx += 1
                
                fixed = False
                for c in range(4):
                    if _has_c6_in_color(color_edges[c], target_n, c):
                        # Find a specific C6
                        for subset in combinations(range(target_n), 6):
                            for cycle in _CYCLES:
                                all_in_color = True
                                for k in range(6):
                                    a = subset[cycle[k]]
                                    b = subset[cycle[(k+1)%6]]
                                    if frozenset((a, b)) not in color_edges[c]:
                                        all_in_color = False
                                        break
                                if all_in_color:
                                    # Found a C6, flip a random edge
                                    k = np.random.randint(6)
                                    a = subset[cycle[k]]
                                    b = subset[cycle[(k+1)%6]]
                                    if a > b:
                                        a, b = b, a
                                    edge_idx = _edge_index(a, b, target_n)
                                    old_color = colors[edge_idx]
                                    new_color = (old_color + 1) % 4  # Cycle through colors
                                    colors[edge_idx] = new_color
                                    # Update color_edges
                                    color_edges[old_color].discard(frozenset((a, b)))
                                    color_edges[new_color].add(frozenset((a, b)))
                                    improved = True
                                    fixed = True
                                    break
                            if fixed:
                                break
                    if fixed:
                        break
                
                if not fixed and not _is_valid_full(colors, target_n):
                    # Stuck, break
                    break
        
        if best_n >= target_n:
            continue
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
