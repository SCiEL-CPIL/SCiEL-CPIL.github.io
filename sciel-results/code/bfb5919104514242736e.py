"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n = 16
    
    # All possible edges for K_16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(all_edges)
    
    # Precompute triangle conflicts: for each edge, which other edges form a triangle with it
    # edge (u,v) conflicts with (u,w) and (v,w) for w != u,v
    edge_index = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    # For each edge, find all edges that would form a triangle if both were included
    triangle_conflicts = [[] for _ in range(num_edges)]
    for idx, (u, v) in enumerate(all_edges):
        # Find all w such that (u,w) and (v,w) exist
        for w in range(n):
            if w == u or w == v:
                continue
            # Check if (u,w) is an edge
            if (u, w) in edge_index:
                ew_idx = edge_index[(u, w)]
                if ew_idx != idx:
                    triangle_conflicts[idx].append(ew_idx)
            if (v, w) in edge_index:
                ew_idx = edge_index[(v, w)]
                if ew_idx != idx:
                    triangle_conflicts[idx].append(ew_idx)
    
    # Remove duplicates in conflict lists
    for i in range(num_edges):
        triangle_conflicts[i] = list(set(triangle_conflicts[i]))
    
    # Helper: check if a set of edges (as a boolean array) forms a triangle-free graph
    def is_triangle_free(edge_set):
        for i in range(num_edges):
            if not edge_set[i]:
                continue
            for j in triangle_conflicts[i]:
                if j > i and edge_set[j]:
                    return False
        return True
    
    # Helper: check if graph with given edges is 3-colorable using backtracking
    def is_3_colorable(edge_set):
        # Build adjacency
        adj = [set() for _ in range(n)]
        for i in range(num_edges):
            if edge_set[i]:
                u, v = all_edges[i]
                adj[u].add(v)
                adj[v].add(u)
        
        color = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            # Try colors 0, 1, 2
            used = set()
            for u in adj[v]:
                if color[u] != -1:
                    used.add(color[u])
            for c in range(3):
                if c not in used:
                    color[v] = c
                    if backtrack(v + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    # Greedy coloring to get a lower bound on chromatic number (for pruning)
    def greedy_chromatic_lower_bound(edge_set):
        adj = [set() for _ in range(n)]
        for i in range(num_edges):
            if edge_set[i]:
                u, v = all_edges[i]
                adj[u].add(v)
                adj[v].add(u)
        
        color = [-1] * n
        max_color = 0
        # Order by degree descending
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        for v in order:
            used = set()
            for u in adj[v]:
                if color[u] != -1:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
            max_color = max(max_color, c)
        return max_color + 1
    
    # Start with the Grötzsch graph (11 edges, 3-colorable... wait, it's not 3-colorable)
    # Actually the Grötzsch graph IS not 3-colorable. Let's use it as a starting point.
    groetzsch_edges = []
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    for u, w in cycle:
        groetzsch_edges.append((u, w))
    for u, w in cycle:
        groetzsch_edges.append((u, 5 + w))
        groetzsch_edges.append((w, 5 + u))
    for u in range(5):
        groetzsch_edges.append((5 + u, 10))
    
    # Convert to edge_set
    best_edge_set = [False] * num_edges
    for e in groetzsch_edges:
        idx = edge_index.get(e)
        if idx is not None:
            best_edge_set[idx] = True
    best_count = sum(best_edge_set)
    
    # Verify it's valid
    if not is_triangle_free(best_edge_set):
        # Shouldn't happen, but reset
        best_edge_set = [False] * num_edges
        best_count = 0
    
    if is_3_colorable(best_edge_set):
        # Shouldn't happen for Grötzsch, but handle it
        best_edge_set = [False] * num_edges
        best_count = 0
    
    # Branch and bound: try to add edges
    # We'll do a systematic search over subsets of the remaining edges
    # Use backtracking with pruning
    
    # Order edges by some heuristic - maybe by how many conflicts they have
    edge_order = sorted(range(num_edges), key=lambda i: len(triangle_conflicts[i]))
    
    # Find edges not in current best
    candidate_edges = [i for i in range(num_edges) if not best_edge_set[i]]
    
    # We'll try adding edges one by one, checking constraints
    # Use a recursive search with time limit
    
    search_state = {
        'edge_set': best_edge_set[:],
        'count': best_count,
        'best_set': best_edge_set[:],
        'best_count': best_count,
        'node_count': 0
    }
    
    def search(idx_in_order, remaining_time):
        if time.time() > deadline - 5:
            return
        
        search_state['node_count'] += 1
        
        if idx_in_order >= len(edge_order):
            return
        
        e_idx = edge_order[idx_in_order]
        
        # Option 1: Don't include this edge
        search(idx_in_order + 1, remaining_time)
        
        # Option 2: Include this edge if it doesn't create a triangle
        if not search_state['edge_set'][e_idx]:
            # Check if adding this edge creates a triangle
            can_add = True
            for conflict in triangle_conflicts[e_idx]:
                if search_state['edge_set'][conflict]:
                    can_add = False
                    break
            
            if can_add:
                search_state['edge_set'][e_idx] = True
                search_state['count'] += 1
                
                # Update best if improved
                if search_state['count'] > search_state['best_count']:
                    # Check if still not 3-colorable
                    if not is_3_colorable(search_state['edge_set']):
                        search_state['best_count'] = search_state['count']
                        search_state['best_set'] = search_state['edge_set'][:]
                
                search(idx_in_order + 1, remaining_time)
                
                search_state['edge_set'][e_idx] = False
                search_state['count'] -= 1
    
    # Do the search
    search(0, 0)
    
    # Use the best found
    final_edge_set = search_state['best_set']
    
    # Convert back to list of edges
    result = []
    for i in range(num_edges):
        if final_edge_set[i]:
            result.append(list(all_edges[i]))
    
    return result
# EVOLVE-BLOCK-END
