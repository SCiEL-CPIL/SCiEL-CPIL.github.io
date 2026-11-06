import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Checks if a 5-uniform hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
        
    # Build adjacency: vertex -> list of edges (indices)
    # And edge -> list of vertices
    # For speed, we work with indices
    edges_list = list(edges)
    m = len(edges_list)
    
    # Precompute vertex to edge index mapping
    vert_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges_list):
        for v in e:
            vert_to_edges[v].append(i)
            
    colors = [-1] * n
    
    # Unit propagation queue
    # We'll implement a simple stack-based propagation
    
    def propagate(start_vertex, start_color):
        # Returns False if contradiction, True if consistent
        # We modify colors in place
        if colors[start_vertex] != -1:
            if colors[start_vertex] != start_color:
                return False
            else:
                return True
        
        colors[start_vertex] = start_color
        stack = [start_vertex]
        
        while stack:
            v = stack.pop()
            for e_idx in vert_to_edges[v]:
                e = edges_list[e_idx]
                c0 = 0
                c1 = 0
                unassigned = []
                for u in e:
                    if colors[u] == 0:
                        c0 += 1
                    elif colors[u] == 1:
                        c1 += 1
                    else:
                        unassigned.append(u)
                
                if c0 == 5 or c1 == 5:
                    return False
                    
                if c0 == 4:
                    # The unassigned vertex must be 1
                    if len(unassigned) == 1:
                        u = unassigned[0]
                        if colors[u] == 0:
                            return False
                        if colors[u] == -1:
                            colors[u] = 1
                            stack.append(u)
                if c1 == 4:
                    # The unassigned vertex must be 0
                    if len(unassigned) == 1:
                        u = unassigned[0]
                        if colors[u] == 1:
                            return False
                        if colors[u] == -1:
                            colors[u] = 0
                            stack.append(u)
                            
        return True

    def backtrack():
        # Find unassigned vertex
        # Heuristic: Pick vertex with most assigned neighbors in edges?
        # For now, simple: first unassigned
        # Better: Pick vertex involved in most edges with unassigned vertices?
        # Let's just pick the first one for simplicity, n is small
        unassigned = -1
        for i in range(n):
            if colors[i] == -1:
                unassigned = i
                break
        
        if unassigned == -1:
            return True
            
        # Try color 0
        saved_colors = colors[:]
        if propagate(unassigned, 0):
            if backtrack():
                return True
            for i in range(n):
                colors[i] = saved_colors[i]
                
        # Try color 1
        saved_colors = colors[:]
        if propagate(unassigned, 1):
            if backtrack():
                return True
            for i in range(n):
                colors[i] = saved_colors[i]
                
        return False

    return backtrack()

def construct():
    best_n = 0
    best_edges = None
    best_count = float('inf')
    
    # Try different n values
    for n in range(9, 13):
        all_edges = [tuple(e) for e in combinations(range(n), 5)]
        num_edges = len(all_edges)
        
        # Run multiple random greedy reductions
        for _ in range(50):
            # Shuffle edge order
            indices = list(range(num_edges))
            random.shuffle(indices)
            
            current_edges = set(all_edges)
            
            # Greedy removal
            for idx in indices:
                e = all_edges[idx]
                if e not in current_edges:
                    continue
                
                # Try removing e
                current_edges.remove(e)
                
                # Check if still non-2-colorable
                # If it becomes 2-colorable, we must restore e
                if is_2_colorable(n, current_edges):
                    current_edges.add(e)
                    # If it was 2-colorable after removal, it means e was critical
                    # We keep e.
                else:
                    # It is still non-2-colorable, so we can keep it removed
                    pass
            
            if len(current_edges) < best_count:
                best_count = len(current_edges)
                best_n = n
                best_edges = list(current_edges)
                
                # Optional: Early stop if we reach a very low number?
                # The prompt says best known is 51.
                # If we find < 51, that's great.
                if best_count <= 28:
                    # The prompt says <= 28 is always 2-colorable, so this shouldn't happen
                    # but if it does, it's a contradiction in the prompt's premise or my check is wrong.
                    # Assuming prompt is correct, we won't see this.
                    pass
                    
    if best_edges is None:
        # Fallback to complete graph on 9 vertices
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': [list(e) for e in best_edges]}
