"""Rewritten program: maximize edges in a triangle-free, non-3-colorable graph on 16 vertices."""
import random
import time
from itertools import combinations

def construct():
    """
    Return a list of edges [u, v] of a simple triangle-free graph on vertices 0..15
    that is not 3-colourable. Maximize the number of edges.
    Uses a randomized local search (hill climbing) starting from a known valid graph
    and trying to add edges while maintaining triangle-free and non-3-colorability.
    """
    n = 16
    
    # Helper: check if adding edge (u, v) creates a triangle
    def creates_triangle(adj_set, u, v):
        # Check if there exists w such that (u,w) and (v,w) are both in adj_set
        common = adj_set[u] & adj_set[v]
        return len(common) > 0
    
    # Helper: exact backtracking 3-colorability check
    def is_3_colorable(adj, n):
        colors = [-1] * n
        # Order vertices by degree descending for efficiency
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Determine available colors
            used = set()
            for u in adj[v]:
                if colors[u] != -1:
                    used.add(colors[u])
            for c in range(3):
                if c not in used:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)
    
    # Helper: build adjacency sets from edge list
    def build_adj_sets(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    # Start with a known valid graph: the Mycielski of C5 (Grötzsch graph) on 11 vertices,
    # plus 5 isolated vertices. This has 15 edges and is triangle-free and not 3-colorable.
    # But we can do better. Let's start with a denser known construction or build one.
    
    # A better starting point: use the Clebsch graph? No, that's 16 vertices, triangle-free,
    # but is it 3-colorable? The Clebsch graph has chromatic number 3, so it IS 3-colorable.
    
    # The Grötzsch graph (11 vertices, 15 edges) is the smallest triangle-free non-3-colorable.
    # We can try to add vertices and edges.
    
    # Let's start with the Grötzsch graph and try to add edges/vertices.
    # Grötzsch graph: vertices 0-9, plus vertex 10.
    # C5: 0-1, 1-2, 2-3, 3-4, 4-0
    # Mycielski: for each edge (u,v) in C5, add (u, 5+v) and (v, 5+u).
    # Then add vertex 10 connected to 5,6,7,8,9.
    
    edges = []
    # C5
    for i in range(5):
        edges.append([i, (i+1) % 5])
    # Mycielski edges
    for i in range(5):
        u, v = i, (i+1) % 5
        edges.append([u, 5 + v])
        edges.append([v, 5 + u])
    # Connect vertex 10 to 5-9
    for i in range(5):
        edges.append([10, 5 + i])
    
    adj = build_adj_sets(edges)
    
    # Verify it's valid
    assert len(edges) == 15
    assert not is_3_colorable(adj, n)
    
    # Now try to add edges. We have vertices 11-15 isolated.
    # Try to add edges incident to these vertices, and also edges among existing vertices
    # that don't create triangles.
    
    # First, let's try to add edges to make the graph denser.
    # We'll use a local search: repeatedly try to add a random edge that doesn't create
    # a triangle, then check if the graph is still non-3-colorable. If yes, keep it.
    # If no, revert.
    
    best_edges = list(edges)
    best_adj = build_adj_sets(best_edges)
    
    # Precompute all possible edges
    all_possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Current edge set for quick lookup
    edge_set = set()
    for u, v in best_edges:
        edge_set.add((min(u,v), max(u,v)))
    
    # Try to add edges one by one
    # Use a greedy approach: for each possible edge, if it doesn't create a triangle,
    # add it and check 3-colorability. If still non-3-colorable, keep it.
    
    # To be more systematic, let's try all possible edges in a random order
    random.seed(42)
    candidate_edges = list(all_possible_edges)
    random.shuffle(candidate_edges)
    
    for u, v in candidate_edges:
        if (u, v) in edge_set or (v, u) in edge_set:
            continue
        # Check if adding this edge creates a triangle
        if creates_triangle(best_adj, u, v):
            continue
        # Try adding the edge
        best_adj[u].add(v)
        best_adj[v].add(u)
        edge_set.add((u, v))
        best_edges.append([u, v])
        
        # Check 3-colorability
        if is_3_colorable(best_adj, n):
            # Revert
            best_adj[u].remove(v)
            best_adj[v].remove(u)
            edge_set.remove((u, v))
            best_edges.pop()
    
    # Now do a local search: try to swap edges or add more
    # Actually, the above greedy might not be optimal. Let's try a more thorough search.
    
    # Let's try simulated annealing style: randomly try to add an edge, if it fails
    # 3-colorability, try removing a different edge to free up space.
    
    # For now, let's also try to add edges among vertices 11-15
    # and edges connecting them to the main graph.
    
    # Let's do another pass with a different order
    random.seed(123)
    candidate_edges2 = list(all_possible_edges)
    random.shuffle(candidate_edges2)
    
    for u, v in candidate_edges2:
        if (u, v) in edge_set or (v, u) in edge_set:
            continue
        if creates_triangle(best_adj, u, v):
            continue
        best_adj[u].add(v)
        best_adj[v].add(u)
        edge_set.add((u, v))
        best_edges.append([u, v])
        
        if is_3_colorable(best_adj, n):
            best_adj[u].remove(v)
            best_adj[v].remove(u)
            edge_set.remove((u, v))
            best_edges.pop()
    
    # Try a few more random restarts with different seeds
    for seed in [456, 789, 101, 202]:
        random.seed(seed)
        # Start from current best
        cur_edges = list(best_edges)
        cur_adj = build_adj_sets(cur_edges)
        cur_edge_set = set((min(u,v), max(u,v)) for u, v in cur_edges)
        
        # Try adding edges in random order
        candidates = list(all_possible_edges)
        random.shuffle(candidates)
        
        for u, v in candidates:
            if (u, v) in cur_edge_set or (v, u) in cur_edge_set:
                continue
            if creates_triangle(cur_adj, u, v):
                continue
            cur_adj[u].add(v)
            cur_adj[v].add(u)
            cur_edge_set.add((u, v))
            cur_edges.append([u, v])
            
            if is_3_colorable(cur_adj, n):
                cur_adj[u].remove(v)
                cur_adj[v].remove(u)
                cur_edge_set.remove((u, v))
                cur_edges.pop()
        
        if len(cur_edges) > len(best_edges):
            best_edges = cur_edges
            best_adj = build_adj_sets(best_edges)
    
    # Final verification
    adj_final = build_adj_sets(best_edges)
    
    # Check triangle-free
    for u in range(n):
        for v in adj_final[u]:
            if u < v:
                for w in adj_final[u]:
                    if w > v and w in adj_final[v]:
                        raise ValueError("Triangle found")
    
    # Check non-3-colorable
    if is_3_colorable(adj_final, n):
        raise ValueError("Graph is 3-colorable")
    
    return best_edges
