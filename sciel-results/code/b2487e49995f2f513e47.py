<<<<<<< SEARCH
import time
import random
import itertools

def is_2_colorable(n, edges):
    # Returns True if 2-colorable, False otherwise.
    # colors: 0 or 1. -1 is unassigned.
    colors = [-1] * n
    # Queue for propagation
    # An edge is a tuple of 5 vertices.
    # We need to check if any edge becomes monochromatic or forces a color.
    
    # To speed up, we can maintain for each vertex a list of edges containing it.
    # But n is small, so iterating all edges might be fine.
    
    # Let's use a recursive backtracking with unit propagation.
    # State: colors array, set of unassigned vertices? No, just colors array.
    
    def propagate(colors):
        # Returns False if contradiction, True if consistent.
        # Modifies colors in place.
        changed = True
        while changed:
            changed = False
            for edge in edges:
                # Count colors in edge
                c0 = 0
                c1 = 0
                c_un = -1
                un_idx = -1
                for v in edge:
                    if colors[v] == 0:
                        c0 += 1
                    elif colors[v] == 1:
                        c1 += 1
                    else:
                        c_un += 1
                        un_idx = v
                
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and c_un == 1:
                    if colors[un_idx] == -1:
                        colors[un_idx] = 1
                        changed = True
                elif c1 == 4 and c_un == 1:
                    if colors[un_idx] == -1:
                        colors[un_idx] = 0
                        changed = True
        return True

    def backtrack(colors):
        if not propagate(colors):
            return False
        
        # Find next unassigned vertex
        next_v = -1
        for v in range(n):
            if colors[v] == -1:
                next_v = v
                break
        
        if next_v == -1:
            return True
        
        # Try color 0
        colors[next_v] = 0
        if backtrack(colors.copy()):
            return True
        
        # Try color 1
        colors[next_v] = 1
        if backtrack(colors.copy()):
            return True
        
        colors[next_v] = -1
        return False

    return backtrack(colors)
=======
import time
import random
import itertools

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation and degree-based vertex ordering.
    """
    if not edges:
        return True
    
    # Build vertex-to-edge index
    v_edges = [[] for _ in range(n)]
    for i, edge in enumerate(edges):
        for v in edge:
            v_edges[v].append(i)
    
    colors = [-1] * n
    edge_c0 = [0] * len(edges)  # count of color-0 in each edge
    edge_c1 = [0] * len(edges)  # count of color-1 in each edge
    edge_un = [5] * len(edges)  # count of unassigned in each edge
    edge_un_vertex = [[-1] * 5 for _ in range(len(edges))]  # unassigned vertices per edge
    
    def propagate():
        """Returns False on contradiction, True otherwise. Modifies colors, edge_c0, edge_c1, edge_un."""
        changed = True
        while changed:
            changed = False
            for ei in range(len(edges)):
                if edge_c0[ei] == 5 or edge_c1[ei] == 5:
                    return False
                if edge_c0[ei] == 4 and edge_un[ei] == 1:
                    # Force the last unassigned vertex to color 1
                    for v in edges[ei]:
                        if colors[v] == -1:
                            colors[v] = 1
                            edge_c1[ei] += 1
                            edge_un[ei] -= 1
                            # Propagate to other edges containing v
                            for oj in v_edges[v]:
                                if oj != ei:
                                    edge_c1[oj] += 1
                                    edge_un[oj] -= 1
                            changed = True
                            break
                elif edge_c1[ei] == 4 and edge_un[ei] == 1:
                    # Force the last unassigned vertex to color 0
                    for v in edges[ei]:
                        if colors[v] == -1:
                            colors[v] = 0
                            edge_c0[ei] += 1
                            edge_un[ei] -= 1
                            # Propagate to other edges containing v
                            for oj in v_edges[v]:
                                if oj != ei:
                                    edge_c0[oj] += 1
                                    edge_un[oj] -= 1
                            changed = True
                            break
        return True
    
    def backtrack():
        if not propagate():
            return False
        
        # Find unassigned vertex with highest degree (most edges)
        next_v = -1
        max_deg = -1
        for v in range(n):
            if colors[v] == -1:
                deg = len(v_edges[v])
                if deg > max_deg:
                    max_deg = deg
                    next_v = v
        
        if next_v == -1:
            return True
        
        # Save state
        saved_colors = colors[:]
        saved_c0 = edge_c0[:]
        saved_c1 = edge_c1[:]
        saved_un = edge_un[:]
        
        # Try color 0
        colors[next_v] = 0
        for oj in v_edges[next_v]:
            edge_c0[oj] += 1
            edge_un[oj] -= 1
        if backtrack():
            return True
        
        # Restore
        colors[:] = saved_colors
        edge_c0[:] = saved_c0
        edge_c1[:] = saved_c1
        edge_un[:] = saved_un
        
        # Try color 1
        colors[next_v] = 1
        for oj in v_edges[next_v]:
            edge_c1[oj] += 1
            edge_un[oj] -= 1
        if backtrack():
            return True
        
        # Restore
        colors[:] = saved_colors
        edge_c0[:] = saved_c0
        edge_c1[:] = saved_c1
        edge_un[:] = saved_un
        
        return False
    
    return backtrack()
>>>>>>> REPLACE

