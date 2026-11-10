<<<<<<< SEARCH
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
=======
"""Find a small 5-uniform hypergraph without Property B (no proper 2-colouring)."""
import time
import random
import sys
from itertools import combinations

def _is_2_colorable(n, edges):
    """
    Fast check for 2-colorability using bitmask backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    m = len(edges)
    
    # Precompute edge masks and vertex lists
    edge_masks = []
    edge_verts = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        edge_verts.append(e)
        
    # Precompute for each vertex, the list of edges it belongs to
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_verts):
        for v in e:
            vertex_edges[v].append(i)
            
    # State:
    # colors: -1 (unassigned), 0, 1
    # edge_counts: [count_0, count_1] for each edge
    
    colors = [-1] * n
    edge_counts = [[0, 0] for _ in range(m)]
    
    sys.setrecursionlimit(10000)
    
    def propagate():
        """
        Apply unit propagation.
        Returns True if consistent, False if contradiction.
        """
        changed = True
        while changed:
            changed = False
            for i in range(m):
                c0, c1 = edge_counts[i]
                # If edge is monochromatic and full, contradiction
                # If edge has 4 of one color and 1 unassigned, force the unassigned to be the other color
                
                # Count unassigned vertices in this edge
                unassigned = 0
                for v in edge_verts[i]:
                    if colors[v] == -1:
                        unassigned += 1
                        
                if unassigned == 0:
                    # Fully assigned
                    if c0 == 0 or c1 == 0:
                        return False
                    continue
                    
                # Unit Propagation:
                # If c0 == 4 and c1 == 0: the remaining 1 must be 1
                # If c1 == 4 and c0 == 0: the remaining 1 must be 0
                # If c0 == 3 and c1 == 1: no constraint (can be 0 or 1? No, if we assign 0, it becomes 4-0 which is bad? 
                # Wait. If we assign 0, it becomes 4 zeros. The last one MUST be 1. 
                # So if c0=3, c1=1, unassigned=1, then the unassigned MUST be 1? 
                # No. If we assign 0 to the last one, we have 4 zeros and 1 one. That is NOT monochromatic. 
                # Monochromatic means ALL 5 are same. 
                # So 4-1 is valid. 
                # The only constraint is: we cannot have 5-0 or 0-5.
                # So if c0=4, c1=0, unassigned=1 -> last MUST be 1.
                # If c1=4, c0=0, unassigned=1 -> last MUST be 0.
                
                if c0 == 4 and c1 == 0:
                    for v in edge_verts[i]:
                        if colors[v] == -1:
                            colors[v] = 1
                            edge_counts[i][1] += 1
                            changed = True
                            break
                elif c1 == 4 and c0 == 0:
                    for v in edge_verts[i]:
                        if colors[v] == -1:
                            colors[v] = 0
                            edge_counts[i][0] += 1
                            changed = True
                            break
                            
        return True

    def backtrack(idx):
        # Find next unassigned vertex
        # Heuristic: pick vertex involved in most "tight" edges? 
        # Simple: pick first unassigned vertex in order of degree
        # To speed up, we can pre-sort vertices by degree
        v = -1
        for i in range(n):
            if colors[i] == -1:
                v = i
                break
                
        if v == -1:
            # All assigned. Check if any edge is monochromatic
            # Propagation should have caught this, but let's verify
            for i in range(m):
                if edge_counts[i][0] == 0 or edge_counts[i][1] == 0:
                    return False
            return True
            
        # Try color 0
        colors[v] = 0
        # Update edge counts
        affected_edges = vertex_edges[v]
        valid = True
        for ei in affected_edges:
            edge_counts[ei][0] += 1
            # Check if this creates a contradiction immediately?
            # If edge_counts[ei][0] == 5, then it's monochromatic 0. Contradiction.
            if edge_counts[ei][0] == 5:
                valid = False
                break
            # Also if edge_counts[ei][1] == 0 and remaining unassigned == 0? 
            # Propagation will handle it.
            
        if valid:
            if propagate():
                if backtrack(idx + 1):
                    return True
            # Undo changes in propagate? 
            # This is tricky. Propagate modifies colors and edge_counts.
            # We need to restore state.
            # Easier: use a stack or copy state. 
            # For n<=26, m<=200, copying arrays is fast.
            # But recursion depth can be 26.
            # Let's try to restore manually.
            # This is error prone. 
            # Alternative: Pass state as arguments? 
            # Or just copy the state before branching.
            
        # Restore state for color 0 attempt
        colors[v] = -1
        for ei in affected_edges:
            edge_counts[ei][0] -= 1
        # We also need to restore changes made by propagate()
        # This is the problem with in-place propagation without undo logs.
        # I will implement a version that copies the state for each branch.
        # Given n is small, this is acceptable.
        
        return False

    # The above backtrack is flawed because propagate modifies state globally.
    # I will rewrite _is_2_colorable to use immutable state passing or careful undo.
    # Given the constraints, a simpler, slightly slower but correct approach is safer.
    # I'll use the "copy state" approach for simplicity and correctness.
    
    def _check_state(colors_copy, edge_counts_copy):
        """
        Recursive check on a specific state.
        """
        # Check for immediate contradiction in edge_counts_copy
        # If any edge is 5-0 or 0-5, return False
        
        # Find unassigned vertex
        v = -1
        # Heuristic: pick vertex with most constraints (highest degree in unassigned graph?)
        # Just pick first one for now, or sort by degree once.
        # Pre-computed vertex_order by degree
        for i in vertex_order:
            if colors_copy[i] == -1:
                v = i
                break
                
        if v == -1:
            # All assigned
            for i in range(m):
                if edge_counts_copy[i][0] == 0 or edge_counts_copy[i][1] == 0:
                    return False
            return True
            
        # Try color 0
        new_colors_0 = colors_copy[:]
        new_counts_0 = [c[:] for c in edge_counts_copy]
        new_colors_0[v] = 0
        valid_0 = True
        for ei in vertex_edges[v]:
            new_counts_0[ei][0] += 1
            if new_counts_0[ei][0] == 5:
                valid_0 = False
                break
                
        if valid_0:
            # Propagate on new state
            if _propagate_state(new_colors_0, new_counts_0):
                if _check_state(new_colors_0, new_counts_0):
                    return True
                    
        # Try color 1
        new_colors_1 = colors_copy[:]
        new_counts_1 = [c[:] for c in edge_counts_copy]
        new_colors_1[v] = 1
        valid_1 = True
        for ei in vertex_edges[v]:
            new_counts_1[ei][1] += 1
            if new_counts_1[ei][1] == 5:
                valid_1 = False
                break
                
        if valid_1:
            # Propagate on new state
            if _propagate_state(new_colors_1, new_counts_1):
                if _check_state(new_colors_1, new_counts_1):
                    return True
                    
        return False
        
    def _propagate_state(colors_arr, edge_counts_arr):
        """
        In-place unit propagation on the passed arrays.
        Returns False if contradiction.
        """
        changed = True
        while changed:
            changed = False
            for i in range(m):
                c0, c1 = edge_counts_arr[i]
                if c0 == 5 or c1 == 5:
                    return False
                    
                # Find unassigned in this edge
                unassigned_v = -1
                for v in edge_verts[i]:
                    if colors_arr[v] == -1:
                        unassigned_v = v
                        break
                        
                if unassigned_v == -1:
                    continue
                    
                # If 4-0, force 1
                if c0 == 4 and c1 == 0:
                    colors_arr[unassigned_v] = 1
                    edge_counts_arr[i][1] += 1
                    changed = True
                elif c1 == 4 and c0 == 0:
                    colors_arr[unassigned_v] = 0
                    edge_counts_arr[i][0] += 1
                    changed = True
        return True
        
    # Precompute vertex order by degree
    deg = [0] * n
    for e in edge_verts:
        for v in e:
            deg[v] += 1
    vertex_order = sorted(range(n), key=lambda v: -deg[v])
    
    initial_colors = [-1] * n
    initial_counts = [[0, 0] for _ in range(m)]
    
    return _check_state(initial_colors, initial_counts)
>>>>>>> REPLACE

This replacement significantly speeds up the 2-colorability check by using unit propagation and efficient state management. This allows the greedy removal in `construct()` to run many more iterations within the time limit, potentially finding a smaller hypergraph.

I will also tweak `construct()` to use the faster checker more aggressively and perhaps try a few different initial n values or random restarts if time permits, but the main gain is from the solver speed.

