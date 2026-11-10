"""Search for dense triangle-free non-3-colorable graphs on 16 vertices."""
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n = 16
    
    def is_triangle_free(adj):
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for w in adj[v]:
                    if w > v and (w in adj[u]):
                        return False
        return True
    
    def is_3_colorable(adj):
        """Backtracking 3-coloring check."""
        color = [-1] * n
        # Order vertices by degree descending for efficiency
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Try each color
            for c in range(3):
                conflict = False
                for nb in adj[v]:
                    if color[nb] == c:
                        conflict = True
                        break
                if not conflict:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    def count_edges(adj):
        total = 0
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    total += 1
        return total
    
    # Start with a known good base: Grötzsch + connections to independent set
    # Grötzsch graph on 0..10
    # Then add vertices 11-15 connected to an independent set
    
    # Independent set in Grötzsch: {0, 1, 2, 3, 4} (the original cycle vertices)
    # Actually, let's use the known construction that gave 45 edges
    
    # Base: Grötzsch graph
    # Cycle 0-1-2-3-4-0
    # Vertex 5 connected to 0,2
    # Vertex 6 connected to 1,3
    # Vertex 7 connected to 2,4
    # Vertex 8 connected to 3,0
    # Vertex 9 connected to 4,1
    # Vertex 10 connected to 5,6,7,8,9
    
    adj = [set() for _ in range(n)]
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
    
    # Grötzsch
    for i in range(5):
        add_edge(i, (i+1) % 5)
    add_edge(5, 0); add_edge(5, 2)
    add_edge(6, 1); add_edge(6, 3)
    add_edge(7, 2); add_edge(7, 4)
    add_edge(8, 3); add_edge(8, 0)
    add_edge(9, 4); add_edge(9, 1)
    for u in range(5, 10):
        add_edge(10, u)
    
    # Independent set in Grötzsch: {0, 1, 2, 3, 4}
    indep = [0, 1, 2, 3, 4]
    
    # Connect vertices 11-15 to the independent set
    for v in range(11, 16):
        for u in indep:
            add_edge(v, u)
    
    # Now try to add more edges via local search
    # Try adding edges between vertices that don't create triangles
    # and check if the graph remains non-3-colorable
    
    best_edges = count_edges(adj)
    best_adj = [set(s) for s in adj]
    
    # List of candidate edges to try adding
    def get_candidate_edges(adj):
        candidates = []
        for u in range(n):
            for v in range(u+1, n):
                if v not in adj[u]:
                    # Check no triangle
                    common = adj[u] & adj[v]
                    if not common:
                        candidates.append((u, v))
        return candidates
    
    # Iteratively try to add edges
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 5:
        improved = False
        candidates = get_candidate_edges(adj)
        # Try each candidate
        for (u, v) in candidates:
            if (time.time() - start_time) >= DEADLINE - 5:
                break
            # Try adding edge
            adj[u].add(v)
            adj[v].add(u)
            
            # Check triangle-free
            triangle = False
            for w in adj[u]:
                if w != v and w in adj[v]:
                    triangle = True
                    break
            if triangle:
                adj[u].discard(v)
                adj[v].discard(u)
                continue
            
            # Check non-3-colorable
            if not is_3_colorable(adj):
                # Valid! Keep this edge
                new_count = count_edges(adj)
                if new_count > best_edges:
                    best_edges = new_count
                    best_adj = [set(s) for s in adj]
                improved = True
            else:
                # Remove edge
                adj[u].discard(v)
                adj[v].discard(u)
    
    # Extract edges from best_adj
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if v > u:
                edges.append([u, v])
    
    return edges
