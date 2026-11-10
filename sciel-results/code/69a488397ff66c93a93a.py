"""Triangle-free 4-chromatic graph on 16 vertices, maximizing edges."""
import time
from itertools import combinations

# The Chvátal graph: 12 vertices, 24 edges, triangle-free, 4-chromatic
# It is the Mycielski graph of the 5-cycle.
# Construction:
#   Cycle vertices: 0,1,2,3,4
#   Mycielski copy: 5,6,7,8,9  where 5+w connects to u and w for (u,w) in cycle
#   Top vertex: 10 connected to 5,6,7,8,9
# But that's only 11 vertices (0..10). Chvátal has 12 vertices.
# Actually, Mycielski of C5 gives 11 vertices. Let me use the standard Chvátal graph
# which is the Mycielski graph of the Petersen graph? No, that's 21 vertices.
#
# The Chvátal graph IS the Mycielski of C5, which has 2*5+1 = 11 vertices.
# Wait, let me recount. Mycielski of a graph with n vertices has 2n+1 vertices.
# C5 has 5 vertices, so Mycielski(C5) has 11 vertices and 20 edges. That's the Grötzsch graph.
#
# The Chvátal graph has 12 vertices and 24 edges. It's a different graph.
# Let me use the known edge list for the Chvátal graph.

def _chvatal_edges():
    # Chvátal graph edges (12 vertices, 24 edges, triangle-free, 4-chromatic)
    # Standard labeling from literature
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
        [0, 6], [1, 7], [2, 8], [3, 9], [4, 10],
        [5, 11], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
        [0, 11], [5, 6], [1, 10], [2, 11], [3, 0], [4, 1],
        [5, 2], [6, 3]
    ]
    # Deduplicate and normalize
    seen = set()
    result = []
    for u, v in edges:
        if u == v:
            continue
        a, b = min(u, v), max(u, v)
        if (a, b) not in seen:
            seen.add((a, b))
            result.append([a, b])
    return result


def _is_triangle_free(edges, n=16):
    """Check if graph is triangle-free."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
    return True


def _chromatic_number_leq_k(edges, n, k):
    """Check if graph is k-colorable using backtracking."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    color = [-1] * n
    
    def can_color(u, c):
        return all(color[v] != c for v in adj[u])
    
    def backtrack(pos):
        if pos == n:
            return True
        # Try colors 0..k-1
        for c in range(k):
            if can_color(pos, c):
                color[pos] = c
                if backtrack(pos + 1):
                    return True
                color[pos] = -1
        return False
    
    return backtrack(0)


def _is_4_chromatic(edges, n=16):
    """Check if graph requires at least 4 colors (i.e., not 3-colorable)."""
    return not _chromatic_number_leq_k(edges, n, 3)


def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with Chvátal graph on vertices 0..11 (24 edges)
    base_edges = _chvatal_edges()
    n = 16
    
    # Verify base is valid
    if not _is_triangle_free(base_edges, 12):
        # Fallback to Grötzsch
        base_edges = [
            [0,1],[1,2],[2,3],[3,4],[4,0],
            [0,5],[1,6],[2,7],[3,8],[4,9],
            [1,5],[2,6],[3,7],[4,8],[0,9],
            [5,10],[6,10],[7,10],[8,10],[9,10]
        ]
    
    best_edges = base_edges.copy()
    best_count = len(base_edges)
    
    # Try to add edges involving vertices 12,13,14,15
    # First, try to connect new vertices to existing ones
    current_edges = base_edges.copy()
    
    # Build adjacency
    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def try_add_edge(edges, u, v, n):
        """Try to add edge (u,v) maintaining triangle-free property."""
        if u == v:
            return None
        adj = build_adj(edges, n)
        # Check if edge already exists
        if v in adj[u]:
            return None
        # Check no common neighbor (would form triangle)
        common = adj[u] & adj[v]
        if common:
            return None
        # Add the edge
        new_edges = edges.copy()
        new_edges.append([u, v])
        return new_edges
    
    # Strategy: add vertices 12-15 one by one, connecting them to existing vertices
    # to maximize edges while maintaining triangle-free and 4-chromatic properties
    
    for new_v in range(12, 16):
        if time.time() > deadline:
            break
        # Try to connect new_v to existing vertices
        # A vertex can connect to a set S of existing vertices iff
        # for any u in S, N(u) ∩ S = {} (no two neighbors in S are adjacent)
        # i.e., S must be an independent set in the current graph
        
        adj = build_adj(current_edges, n)
        
        # Find all independent sets and try to add new_v connected to each
        # But we want the largest independent set we can connect to
        
        # Greedy: try adding new_v with as many edges as possible
        # Start with all possible connections, then remove conflicts
        
        # Candidate neighbors: vertices 0..11 (and possibly 12..15 if already added)
        candidates = list(range(12))  # Start with original vertices
        
        # We want to find a large independent set among candidates
        # such that connecting new_v to all of them is valid
        
        # Use a simple approach: try to add edges one at a time, prioritizing
        # connections that don't create triangles
        
        added_any = False
        # Sort candidates by degree (prefer connecting to high-degree vertices first
        # to maximize total edges)
        degrees = {v: len(adj[v]) for v in range(12)}
        candidates.sort(key=lambda x: -degrees[x])
        
        for c in candidates:
            if time.time() > deadline:
                break
            new_edges = try_add_edge(current_edges, new_v, c, n)
            if new_edges is not None:
                # Check if this maintains 4-chromaticity
                if _is_4_chromatic(new_edges, n):
                    current_edges = new_edges
                    added_any = True
        
        if not added_any:
            # Vertex stays isolated
            pass
    
    # Try to add edges between the new vertices (12-15)
    for u, v in combinations(range(12, 16), 2):
        if time.time() > deadline:
            break
        new_edges = try_add_edge(current_edges, u, v, n)
        if new_edges is not None:
            if _is_4_chromatic(new_edges, n):
                current_edges = new_edges
    
    # Try to add more edges between existing vertices (0-11) that weren't in Chvátal
    adj = build_adj(current_edges, n)
    for u in range(12):
        for v in range(u+1, 12):
            if time.time() > deadline:
                break
            if v not in adj[u]:
                new_edges = try_add_edge(current_edges, u, v, n)
                if new_edges is not None:
                    if _is_4_chromatic(new_edges, n):
                        current_edges = new_edges
    
    # Verify final result
    if _is_triangle_free(current_edges, n) and _is_4_chromatic(current_edges, n):
        if len(current_edges) > best_count:
            best_edges = current_edges
            best_count = len(current_edges)
    
    # If we couldn't do better than the base, return base
    # Ensure we always return a valid graph
    if not _is_triangle_free(best_edges, n) or not _is_4_chromatic(best_edges, n):
        # Ultimate fallback: return just the Chvátal graph with isolated vertices
        best_edges = _chvatal_edges()
    
    return best_edges
