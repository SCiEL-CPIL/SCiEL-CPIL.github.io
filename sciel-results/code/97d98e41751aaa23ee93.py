"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    """
    Construct a triangle-free graph on vertices 0..15 that is not 3-colorable,
    maximizing the number of edges.
    
    Strategy:
    1. Start with a known triangle-free non-3-colorable graph on 11 vertices (Grötzsch graph)
       and 5 isolated vertices.
    2. Use local search to add edges while maintaining triangle-freeness and non-3-colorability.
    3. Also try starting from other bases and different random initializations.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    best_edges = None
    best_count = -1
    
    def is_triangle_free(edges_set, n=16):
        """Check if graph is triangle-free."""
        # Build adjacency for quick lookup
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check all triangles
        for u in range(n):
            for v in list(adj[u]):
                if v > u:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            return False
        return True
    
    def is_not_3_colorable(edges_list, n=16):
        """
        Check if graph is NOT 3-colorable using backtracking.
        Returns True if the graph requires 4 or more colors.
        """
        adj = [set() for _ in range(n)]
        for u, v in edges_list:
            adj[u].add(v)
            adj[v].add(u)
        
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        pos = {v: i for i, v in enumerate(order)}
        
        color = [-1] * n
        colors = [0, 1, 2]
        
        # Check if a vertex can be assigned a color
        def can_assign(v, c):
            for nb in adj[v]:
                if color[nb] == c:
                    return False
            return True
        
        # Backtracking search
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Try each color
            for c in colors:
                if can_assign(v, c):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return not backtrack(0)
    
    def edge_count(edges_list):
        return len(edges_list)
    
    def get_edges_set(edges_list):
        return frozenset(frozenset(e) for e in edges_list)
    
    # Grötzsch graph on vertices 0..10
    def make_grotzsch():
        edges = []
        # Cycle 0-1-2-3-4-0
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        for u, v in cycle:
            edges.append([u, v])
        # Mycielski: add vertices 5..9 connected to cycle
        for u, w in cycle:
            edges.append([u, 5 + w])
            edges.append([w, 5 + u])
        # Add vertex 10 connected to 5..9
        for u in range(5):
            edges.append([5 + u, 10])
        return edges
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Start from Grötzsch + 5 isolated
    grotzsch_edges = make_grotzsch()
    strategies.append(grotzsch_edges[:])
    
    # Strategy 2: Start from a random triangle-free graph and try to make it non-3-colorable
    # This is harder, so we'll focus on growing from Grötzsch
    
    # Local search: add edges one at a time
    def local_search(initial_edges, max_iter=100000):
        """Try to add edges while maintaining constraints."""
        edges = initial_edges[:]
        edges_set = get_edges_set(edges)
        n = 16
        
        # Precompute all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Current state
        is_tf = is_triangle_free(edges_set)
        is_not3c = is_not_3_colorable(edges)
        
        if not is_tf:
            return edges
        if not is_not3c:
            return edges
        
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Check time
            if time.time() > deadline:
                break
            
            # Shuffle candidate edges for diversity
            candidates = [e for e in all_edges if frozenset(e) not in edges_set]
            random.shuffle(candidates)
            
            for u, v in candidates:
                if time.time() > deadline:
                    break
                
                # Check if adding edge (u,v) creates a triangle
                # Triangle exists if there's a w adjacent to both u and v
                adj_u = set()
                adj_v = set()
                for a, b in edges:
                    if a == u:
                        adj_u.add(b)
                    elif b == u:
                        adj_u.add(a)
                    if a == v:
                        adj_v.add(b)
                    elif b == v:
                        adj_v.add(a)
                
                if adj_u & adj_v:
                    continue  # Would create a triangle
                
                # Try adding the edge
                new_edges = edges + [[u, v]]
                new_edges_set = edges_set | frozenset(frozenset((u, v)))
                
                # Check non-3-colorability
                if is_not_3_colorable(new_edges):
                    # Accept
                    edges = new_edges
                    edges_set = new_edges_set
                    improved = True
                    break
        
        return edges
    
    # Run local search from Grötzsch
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        # Add some random edges initially to diversify
        base = grotzsch_edges[:]
        
        # Try adding random edges that don't create triangles
        adj = [set() for _ in range(16)]
        for u, v in base:
            adj[u].add(v)
            adj[v].add(u)
        
        # Try to add random edges
        for _ in range(20):
            u = random.randint(0, 15)
            v = random.randint(0, 15)
            if u == v:
                continue
            if v in adj[u]:
                continue
            # Check triangle
            if adj[u] & adj[v]:
                continue
            base.append([u, v])
            adj[u].add(v)
            adj[v].add(u)
        
        if not is_triangle_free(get_edges_set(base)):
            continue
        if not is_not_3_colorable(base):
            continue
        
        result = local_search(base)
        if len(result) > best_count:
            best_count = len(result)
            best_edges = result[:]
    
    # Also try a more systematic approach: start from Grötzsch and greedily add edges
    # in order of potential (vertices with more freedom first)
    if time.time() < deadline:
        base = grotzsch_edges[:]
        adj = [set() for _ in range(16)]
        for u, v in base:
            adj[u].add(v)
            adj[v].add(u)
        
        # Compute degrees
        deg = [len(adj[i]) for i in range(16)]
        
        # Try to add edges between low-degree vertices first
        candidates = []
        for i in range(16):
            for j in range(i + 1, 16):
                if j not in adj[i]:
                    candidates.append((i, j))
        
        # Sort by sum of degrees (prefer low-degree vertices)
        candidates.sort(key=lambda x: deg[x[0]] + deg[x[1]])
        
        for u, v in candidates:
            if time.time() > deadline:
                break
            
            if v in adj[u]:
                continue
            
            # Check triangle
            if adj[u] & adj[v]:
                continue
            
            new_edges = base + [[u, v]]
            if is_not_3_colorable(new_edges):
                base = new_edges
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
        
        if len(base) > best_count:
            best_count = len(base)
            best_edges = base[:]
    
    # If we still have time, try more local search iterations
    if best_edges and time.time() < deadline:
        for _ in range(5):
            if time.time() > deadline:
                break
            result = local_search(best_edges[:], max_iter=50000)
            if len(result) > best_count:
                best_count = len(result)
                best_edges = result[:]
    
    if best_edges is None:
        # Fallback: return Grötzsch graph
        return grotzsch_edges
    
    return best_edges

# EVOLVE-BLOCK-END
