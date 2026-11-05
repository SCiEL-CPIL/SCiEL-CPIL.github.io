"""Find maximum-edge triangle-free non-3-colorable graph on 16 vertices."""
import time
from itertools import combinations

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n = 16
    
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Precompute adjacency for triangle check
    # We'll build the graph incrementally
    
    best_edges = []
    best_count = 0
    
    # Start with a known good solution: Mycielski of C5 (Grötzsch graph) on vertices 0-10
    # vertices 11-15 isolated
    def groetzsch_edges():
        edges = []
        # C5: 0-1-2-3-4-0
        for i in range(5):
            edges.append([i, (i+1)%5])
        # Mycielski: for each edge (u,v) in C5, add u'-(v) and v'-(u) where u'=5+u
        for i in range(5):
            u = i
            v = (i+1)%5
            edges.append([5+u, v])
            edges.append([5+v, u])
        # Add vertex 10 connected to all 5',6',7',8',9'
        for i in range(5):
            edges.append([10, 5+i])
        return edges
    
    initial = groetzsch_edges()
    best_edges = initial[:]
    best_count = len(initial)
    
    # Build adjacency set for quick lookup
    adj = set()
    for e in best_edges:
        adj.add((e[0], e[1]))
        adj.add((e[1], e[0]))
    
    # Function to check if adding edge (u,v) creates a triangle
    def creates_triangle(u, v):
        for w in range(n):
            if w != u and w != v:
                if (u, w) in adj and (v, w) in adj:
                    return True
        return False
    
    # Function to check 3-colorability (returns True if 3-colorable)
    def is_3_colorable(edge_list):
        # Build adjacency list
        adj_list = [[] for _ in range(n)]
        for u, v in edge_list:
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        colors = [-1] * n
        
        def backtrack(vertex):
            if vertex == n:
                return True
            # Try colors 0, 1, 2
            for c in range(3):
                valid = True
                for nb in adj_list[vertex]:
                    if colors[nb] == c:
                        valid = False
                        break
                if valid:
                    colors[vertex] = c
                    if backtrack(vertex+1):
                        return True
                    colors[vertex] = -1
            return False
        
        return backtrack(0)
    
    # Candidate edges to try adding (those not in current graph)
    current_edge_set = set()
    for e in best_edges:
        current_edge_set.add((min(e[0],e[1]), max(e[0],e[1])))
    
    candidates = [(u,v) for (u,v) in all_edges if (u,v) not in current_edge_set]
    
    # Sort candidates by potential: prefer edges between high-degree vertices or that are "dense"
    # Actually, let's try all candidates in order, using backtracking
    
    # Use a smarter approach: iterative deepening on number of additional edges
    # Try to add k edges at a time, for k=1,2,3,...
    
    def search_add_edges(current_edges, current_adj_set, current_adj_list, max_additional, deadline):
        """Try to add up to max_additional more edges while maintaining triangle-free and non-3-colorable."""
        if time.time() > deadline:
            return None
        
        if max_additional == 0:
            return current_edges[:]
        
        # Try each candidate edge
        for u, v in all_edges:
            if (u,v) in current_adj_set or (v,u) in current_adj_set:
                continue
            # Check triangle
            if creates_triangle_with_set(u, v, current_adj_set, n):
                continue
            
            # Add edge
            new_edges = current_edges + [[u, v]]
            new_adj_set = current_adj_set | {(u,v), (v,u)}
            
            # Check if still not 3-colorable
            if is_3_colorable(new_edges):
                continue
            
            # Recurse
            result = search_add_edges(new_edges, new_adj_set, None, max_additional - 1, deadline)
            if result is not None and len(result) > len(current_edges):
                return result
            
            # Remove edge (not needed since we passed by value)
        
        return current_edges[:]
    
    def creates_triangle_with_set(u, v, adj_set, n):
        for w in range(n):
            if w != u and w != v:
                if (u, w) in adj_set and (v, w) in adj_set:
                    return True
        return False
    
    # Try adding 1 edge at a time, greedily
    for attempt in range(1, 6):  # try to add up to 5 more edges
        if time.time() > deadline:
            break
        
        # Reset to best known
        current_edges = best_edges[:]
        current_adj_set = set()
        for e in current_edges:
            current_adj_set.add((e[0], e[1]))
            current_adj_set.add((e[1], e[0]))
        
        # Greedily try to add edges
        found = True
        while found and time.time() < deadline:
            found = False
            for u, v in all_edges:
                if (u,v) in current_adj_set or (v,u) in current_adj_set:
                    continue
                if creates_triangle_with_set(u, v, current_adj_set, n):
                    continue
                
                new_edges = current_edges + [[u, v]]
                new_adj_set = current_adj_set | {(u,v), (v,u)}
                
                # Check non-3-colorable
                if is_3_colorable(new_edges):
                    continue
                
                # Accept this edge
                current_edges = new_edges
                current_adj_set = new_adj_set
                found = True
                break
        
        if len(current_edges) > best_count:
            best_edges = current_edges[:]
            best_count = len(current_edges)
    
    return best_edges
