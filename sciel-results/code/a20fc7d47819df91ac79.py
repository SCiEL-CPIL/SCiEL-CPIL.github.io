import time
import random

def _is_triangle_free(adj, n):
    """Check if graph with adjacency matrix adj on n vertices is triangle-free."""
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                for k in range(j + 1, n):
                    if adj[i][k] and adj[j][k]:
                        return False
    return True

def _is_3_colorable(adj, n):
    """Backtracking 3-coloring check."""
    color = [-1] * n
    
    # Order vertices by degree (descending)
    deg = [sum(adj[v]) for v in range(n)]
    order = sorted(range(n), key=lambda v: -deg[v])
    pos_in_order = [0] * n
    for i, v in enumerate(order):
        pos_in_order[v] = i
    
    def bt(idx):
        if idx == n:
            return True
        v = order[idx]
        # Determine available colors
        used = set()
        for u in range(n):
            if adj[v][u] and color[u] != -1:
                used.add(color[u])
        for c in range(3):
            if c not in used:
                color[v] = c
                if bt(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return bt(0)

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s budget, 20s safety margin
    
    n = 16
    total_possible_edges = n * (n - 1) // 2
    # List of all possible edges (u, v) with u < v
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Start with Grötzsch graph edges (known valid, 20 edges)
    groetzsch_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [0, 5], [5, 1], [1, 6], [6, 2], [2, 7], [7, 3], [3, 8], [8, 4], [4, 9], [9, 0],
        [5, 10], [6, 10], [7, 10], [8, 10], [9, 10]
    ]
    
    best_edges = groetzsch_edges
    best_count = len(groetzsch_edges)
    
    # Build adjacency
    adj = [[False] * n for _ in range(n)]
    for u, v in groetzsch_edges:
        adj[u][v] = adj[v][u] = True
    
    # Candidate edges to try adding (vertices 11-15 are isolated in Groetzsch)
    # First, let's try to build a better graph using local search
    
    # Strategy: Use simulated annealing or hill climbing on the edge set
    # State: set of edges. Objective: maximize edges subject to triangle-free and not 3-colorable
    
    # Let's use a simpler approach: start from Groetzsch and greedily add edges
    # then try to swap edges to add more
    
    # First, let's try a constructive approach:
    # We know that for triangle-free graphs, the Mycielski construction preserves chromatic number
    # but increases vertices. Let's try to build a denser graph.
    
    # Actually, let's use random local search:
    # 1. Start with a random triangle-free graph that is not 3-colorable
    # 2. Try to add random edges, if it creates a triangle, remove one of the edges
    # 3. Try to remove and add different edges to increase total edge count
    
    random.seed(42)
    
    # Better approach: use the known fact that we can construct denser graphs
    # Let's try to use a bipartite graph as a base and add a few edges to make it non-3-colorable
    
    # A complete bipartite graph K_{8,8} has 64 edges and is 2-colorable (hence 3-colorable)
    # If we add edges within one part, we get triangles
    # So we need a different approach
    
    # Let's use the following construction:
    # Take C5 (5-cycle), its Mycielski is the 11-vertex Grötzsch graph
    # The Mycielski of Grötzsch would be 21 vertices, too many
    
    # Alternative: use a known dense triangle-free non-3-colorable graph
    # The Clebsch graph (16 vertices, 30 edges, triangle-free, 4-chromatic)
    
    # Let's construct the Clebsch graph!
    # The Clebsch graph is the folded 5-cube, or the Kneser graph KG(5,2)
    # Vertices: all 2-element subsets of {0,1,2,3,4}
    # Edge: two vertices are adjacent if they are disjoint
    
    clebsch_edges = []
    vertices = []
    for i in range(5):
        for j in range(i + 1, 5):
            vertices.append((i, j))
    
    # Map (i,j) to vertex index 0..9
    vertex_to_idx = {}
    for idx, v in enumerate(vertices):
        vertex_to_idx[v] = idx
    
    for idx1, v1 in enumerate(vertices):
        for idx2, v2 in enumerate(vertices):
            if idx1 < idx2:
                # Check if disjoint
                if len(set(v1) & set(v2)) == 0:
                    clebsch_edges.append([idx1, idx2])
    
    # Clebsch graph has 10 vertices and 30 edges
    # It's triangle-free and 4-chromatic
    
    # Now we need to extend to 16 vertices
    # Add 6 more vertices. To maintain triangle-free and non-3-colorable:
    # We can add isolated vertices, but that doesn't help
    # Or we can connect them carefully
    
    # Actually, the Clebsch graph itself on 10 vertices with 30 edges is already good
    # But we need 16 vertices. Let's see if we can do better.
    
    # What if we take two disjoint copies of a smaller graph?
    # Two disjoint Grötzsch graphs: 22 vertices, too many
    # One Grötzsch (11) + 5 isolated: 20 edges
    
    # Let's try: Clebsch (10 vertices, 30 edges) + 6 more vertices
    # If we can connect the 6 new vertices to independent sets in Clebsch, we can add edges
    
    # Independence number of Clebsch: since it's 4-chromatic and has 10 vertices,
    # alpha <= 10/2 = 5 (by a simple bound, but let's check)
    # Actually, for the Clebsch graph, alpha = 4 (I believe)
    
    # Let's just use the Clebsch graph plus 6 isolated vertices for now
    # That gives 30 edges, which is better than 20
    
    # But can we do better? Let's try to add edges from the 6 new vertices
    
    # Construct the full 16-vertex graph
    adj16 = [[False] * n for _ in range(n)]
    for u, v in clebsch_edges:
        adj16[u][v] = adj16[v][u] = True
    
    # Now try to add edges involving vertices 10..15
    # For each new vertex, try to connect it to an independent set in the existing graph
    
    # First, find independent sets in the Clebsch graph (vertices 0..9)
    # An independent set in Clebsch is a set of pairwise non-adjacent vertices
    
    def find_max_independent_set(adj_sub, vertices_sub):
        """Find a large independent set in the subgraph induced by vertices_sub."""
        best = []
        
        def backtrack(remaining, current):
            nonlocal best
            if not remaining:
                if len(current) > len(best):
                    best = current[:]
                return
            # Try adding the first vertex
            v = remaining[0]
            rest = remaining[1:]
            
            # Option 1: include v
            new_remaining = [u for u in rest if not adj_sub[v][u]]
            current.append(v)
            backtrack(new_remaining, current)
            current.pop()
            
            # Option 2: exclude v
            backtrack(rest, current)
        
        backtrack(list(vertices_sub), [])
        return best
    
    # Find maximum independent sets in Clebsch graph
    clebsch_vertices = list(range(10))
    max_iso = find_max_independent_set(adj16, clebsch_vertices)
    
    # Add edges from vertices 10..15 to max_iso
    for v in range(10, 16):
        for u in max_iso:
            adj16[v][u] = adj16[u][v] = True
    
    # Convert back to edge list
    edges16 = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj16[u][v]:
                edges16.append([u, v])
    
    # Validate
    if _is_triangle_free(adj16, n) and not _is_3_colorable(adj16, n):
        if len(edges16) > best_count:
            best_edges = edges16
            best_count = len(edges16)
    
    # Now try local search to improve
    # Try adding random edges and removing edges that create triangles
    
    current_adj = [row[:] for row in adj16]
    current_edges = edges16[:]
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Try to add a random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v or current_adj[u][v]:
            continue
        
        # Check if adding (u,v) creates a triangle
        creates_triangle = False
        for w in range(n):
            if w != u and w != v and current_adj[u][w] and current_adj[v][w]:
                creates_triangle = True
                break
        
        if not creates_triangle:
            # Add the edge
            current_adj[u][v] = current_adj[v][u] = True
            current_edges.append([u, v])
            
            # Check if still not 3-colorable
            if not _is_3_colorable(current_adj, n):
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = current_edges[:]
            else:
                # Revert: the graph became 3-colorable
                current_adj[u][v] = current_adj[v][u] = False
                current_edges.pop()
        else:
            # Try to remove one of the edges that creates the triangle and add the new one
            # This is a 2-opt move
            for w in range(n):
                if w != u and w != v and current_adj[u][w] and current_adj[v][w]:
                    # Consider removing (u,w) or (v,w) and adding (u,v)
                    # Remove (u,w)
                    current_adj[u][w] = current_adj[w][u] = False
                    current_edges.remove([u, w] if u < w else [w, u])
                    current_adj[u][v] = current_adj[v][u] = True
                    current_edges.append([u, v])
                    
                    if _is_triangle_free(current_adj, n) and not _is_3_colorable(current_adj, n):
                        if len(current_edges) > best_count:
                            best_count = len(current_edges)
                            best_edges = current_edges[:]
                    else:
                        # Revert
                        current_adj[u][v] = current_adj[v][u] = False
                        current_edges.remove([u, v])
                        current_adj[u][w] = current_adj[w][u] = True
                        current_edges.append([u, w] if u < w else [w, u])
                    
                    if time.time() > deadline:
                        break
                    break
    
    # Final validation
    final_adj = [[False] * n for _ in range(n)]
    for u, v in best_edges:
        final_adj[u][v] = final_adj[v][u] = True
    
    if not _is_triangle_free(final_adj, n) or _is_3_colorable(final_adj, n):
        # Fallback to known valid Grötzsch graph
        return groetzsch_edges
    
    return best_edges
