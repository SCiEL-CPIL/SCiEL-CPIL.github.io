import time
import random
from itertools import combinations

def construct():
    """
    Find a triangle-free graph on 16 vertices with chromatic number >= 4,
    maximizing the number of edges.
    
    Returns:
        list of edges [u, v]
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    best_edges = None
    best_score = -1
    
    # Known small triangle-free 4-chromatic graphs
    # Grotzsch graph: 11 vertices, 20 edges, triangle-free, 4-chromatic
    grotzsch_edges = []
    # Cycle 0-1-2-3-4-0
    for i in range(5):
        grotzsch_edges.append([i, (i+1)%5])
    # Mycielski construction: vertex 5 connected to 0,4 (neighbors of 1)
    # Actually, let's use the standard Grotzsch graph construction
    # It's the Mycielski graph of C5
    # C5: 0-1-2-3-4-0
    # New vertices: 5,6,7,8,9,10
    # 5 connected to 0,4 (non-neighbor of 1? No, Mycielski: 
    # For each vertex i in C5, add i' connected to all neighbors of i in C5
    # Wait, Mycielski: 
    # Original vertices 0..4
    # Add vertices 5..9 where i+5 is connected to all neighbors of i in C5
    # Add vertex 10 connected to 5..9
    
    # C5 neighbors:
    # 0: 1,4
    # 1: 0,2
    # 2: 1,3
    # 3: 2,4
    # 4: 3,0
    
    grotzsch_edges = []
    # Original C5
    for i in range(5):
        grotzsch_edges.append([i, (i+1)%5])
    # Mycielski vertices 5..9
    # 5 (corresponding to 0) connected to neighbors of 0: 1,4
    grotzsch_edges.append([5, 1])
    grotzsch_edges.append([5, 4])
    # 6 (corresponding to 1) connected to neighbors of 1: 0,2
    grotzsch_edges.append([6, 0])
    grotzsch_edges.append([6, 2])
    # 7 (corresponding to 2) connected to neighbors of 2: 1,3
    grotzsch_edges.append([7, 1])
    grotzsch_edges.append([7, 3])
    # 8 (corresponding to 3) connected to neighbors of 3: 2,4
    grotzsch_edges.append([8, 2])
    grotzsch_edges.append([8, 4])
    # 9 (corresponding to 4) connected to neighbors of 4: 3,0
    grotzsch_edges.append([9, 3])
    grotzsch_edges.append([9, 0])
    # 10 connected to 5,6,7,8,9
    for i in range(5):
        grotzsch_edges.append([10, 5+i])
    
    # That's 5 + 10 + 5 = 20 edges, 11 vertices
    # Add 5 more isolated vertices 11-15
    # But we want to maximize edges, so let's try to add more
    
    # Try to extend the Grotzsch graph
    # The Grotzsch graph is 4-chromatic and triangle-free
    # We have 5 more vertices to place
    
    # Let's use a more systematic approach: random search with local optimization
    
    def is_triangle_free(adj):
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True
    
    def chromatic_number_lower_bound(adj):
        """
        Try to find a 3-coloring. If we fail, the graph is 4-chromatic.
        Returns True if 3-colorable, False otherwise.
        """
        # Backtracking 3-coloring
        color = [-1] * n
        uncolored = list(range(n))
        
        def can_color():
            if not uncolored:
                return True
            
            # Pick uncolored vertex with fewest available colors
            best_idx = -1
            best_count = 100
            for idx, v in enumerate(uncolored):
                available = set([0,1,2])
                for u in range(n):
                    if adj[u][v] and color[u] != -1:
                        available.discard(color[u])
                count = len(available)
                if count == 0:
                    return False
                if count < best_count:
                    best_count = count
                    best_idx = idx
            
            v = uncolored[best_idx]
            uncolored.remove(v)
            available = set([0,1,2])
            for u in range(n):
                if adj[u][v] and color[u] != -1:
                    available.discard(color[u])
            
            for c in available:
                color[v] = c
                if can_color():
                    return True
                color[v] = -1
            
            uncolored.insert(best_idx, v)
            return False
        
        return can_color()
    
    def edge_count(adj):
        return sum(adj[i][j] for i in range(n) for j in range(i+1, n))
    
    # Start with Grotzsch + 5 isolated vertices
    base_adj = [[0]*n for _ in range(n)]
    for u, v in grotzsch_edges:
        base_adj[u][v] = 1
        base_adj[v][u] = 1
    
    # Try to add edges by placing the 5 isolated vertices optimally
    # But first, let's do random search
    
    random.seed(42)
    
    def random_graph():
        adj = [[0]*n for _ in range(n)]
        # Start with some structure
        # Random triangle-free graph generation
        # Use the fact that triangle-free graphs can be constructed via bipartite + modifications
        # Or just random edges and check triangle-freeness
        
        # Better: start empty, add edges randomly, reject if creates triangle
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.4:
                    # Check if adding (i,j) creates a triangle
                    ok = True
                    for k in range(n):
                        if k != i and k != j:
                            if adj[i][k] and adj[j][k]:
                                ok = False
                                break
                    if ok:
                        adj[i][j] = 1
                        adj[j][i] = 1
        return adj
    
    def local_search(adj):
        """
        Try to improve by adding/removing edges while maintaining
        triangle-freeness and 4-chromaticity.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        # Try removing edge
                        adj[i][j] = 0
                        adj[j][i] = 0
                        if is_triangle_free(adj) and not chromatic_number_lower_bound(adj):
                            # Still 4-chromatic, but fewer edges - don't keep
                            adj[i][j] = 1
                            adj[j][i] = 1
                        else:
                            # Removed edge, check if still valid
                            # If removing edge breaks 4-chromaticity, revert
                            if chromatic_number_lower_bound(adj):
                                adj[i][j] = 1
                                adj[j][i] = 1
                            else:
                                improved = True
                                break
                    else:
                        # Try adding edge
                        ok = True
                        for k in range(n):
                            if k != i and k != j:
                                if adj[i][k] and adj[j][k]:
                                    ok = False
                                    break
                        if ok:
                            adj[i][j] = 1
                            adj[j][i] = 1
                            if not chromatic_number_lower_bound(adj):
                                # Still 4-chromatic, accept
                                improved = True
                                break
                            else:
                                # Adding edge made it 3-colorable, revert
                                adj[i][j] = 0
                                adj[j][i] = 0
                if improved:
                    break
        return adj
    
    # Run multiple random restarts
    num_restarts = 100
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Mix of strategies
        if restart < 30:
            # Start from Grotzsch + isolated
            adj = [row[:] for row in base_adj]
        elif restart < 60:
            # Random graph
            adj = random_graph()
        else:
            # Random graph with higher density
            adj = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        ok = True
                        for k in range(n):
                            if k != i and k != j:
                                if adj[i][k] and adj[j][k]:
                                    ok = False
                                    break
                        if ok:
                            adj[i][j] = 1
                            adj[j][i] = 1
        
        if not is_triangle_free(adj):
            continue
        
        # Check if 4-chromatic
        if chromatic_number_lower_bound(adj):
            continue
        
        # Local search to improve
        adj = local_search(adj)
        
        if not is_triangle_free(adj) or chromatic_number_lower_bound(adj):
            continue
        
        m = edge_count(adj)
        if m > best_score:
            best_score = m
            best_adj = [row[:] for row in adj]
    
    if best_adj is not None:
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if best_adj[i][j]:
                    edges.append([i, j])
        return edges
    
    # Fallback: return Grotzsch + isolated
    edges = []
    for u, v in grotzsch_edges:
        edges.append([u, v])
    return edges
